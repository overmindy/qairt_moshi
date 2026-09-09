"""Patch and validate dynamic RMSNorm in the real layers 6:8 ONNX shard."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper
import onnxruntime as ort

from moshi_verify_cloud_chain import cloud_step, metrics, save_json


def prepare(args: argparse.Namespace) -> None:
    if any(args.output_dir.iterdir()):
        raise ValueError("Use an empty output directory")
    model = onnx.load(str(args.export_dir / "temporal_layers_6_7.onnx"))
    constants = {value.name: numpy_helper.to_array(value) for value in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type == "Constant":
            for attribute in node.attribute:
                if attribute.name == "value":
                    constants[node.output[0]] = numpy_helper.to_array(attribute.t)
    producers = {output: node for node in model.graph.node for output in node.output}
    reductions = [node for node in model.graph.node if node.op_type == "ReduceMean"
                  and ("/norm1/" in node.name or "/norm2/" in node.name)]
    if len(reductions) != 4:
        raise ValueError(f"Expected four RMSNorms, found {len(reductions)}")
    insertions = {}
    changes = []
    for index, reduction in enumerate(reductions):
        prefix = reduction.name.rsplit("/", 1)[0] + "/"
        square = producers[reduction.input[0]]
        if square.op_type not in ("Pow", "Mul"):
            raise ValueError(f"Unexpected square: {square.name}")
        operand = square.input[0]
        if square.op_type == "Mul" and square.input[1] != operand:
            raise ValueError("Expected elementwise square")
        if square.op_type == "Pow":
            exponent = constants.get(square.input[1])
            if exponent is None or exponent.size != 1 or float(exponent.item()) != 2:
                raise ValueError("Expected a constant exponent of two")
        additions = [node for node in model.graph.node if node.op_type == "Add"
                     and reduction.output[0] in node.input]
        if len(additions) != 1:
            raise ValueError("Expected unique epsilon addition")
        addition = additions[0]
        epsilon = next(name for name in addition.input if name != reduction.output[0])
        epsilon_value = constants.get(epsilon)
        if epsilon_value is None or epsilon_value.size != 1 or not 0 < float(epsilon_value.item()) < 0.01:
            raise ValueError("Expected a positive scalar RMSNorm epsilon")
        tag = f"dynamic_norm_{index}"
        members = [node for node in model.graph.node if node.name.startswith(prefix)]
        consumers = [node for node in members if operand in node.input]
        if not any(node.name != square.name for node in consumers):
            raise ValueError("Cannot identify normalization numerator")
        first = next(node for node in members if operand in node.input)
        inserted = [
            helper.make_node("Abs", [operand], [tag + "_abs"]),
            helper.make_node("ReduceMax", [tag + "_abs"], [tag + "_max"], axes=[-1], keepdims=1),
            helper.make_node("Max", [tag + "_max", "dynamic_one"], [tag + "_scale"]),
            helper.make_node("Div", [operand, tag + "_scale"], [tag + "_input"]),
        ]
        for offset, node in enumerate(inserted):
            node.name = f"{tag}_input_{offset}"
        insertions.setdefault(first.name, []).extend(inserted)
        insertions.setdefault(addition.name, []).extend([
            helper.make_node("Div", [epsilon, tag + "_scale"], [tag + "_eps_once"], name=tag + "_eps1"),
            helper.make_node("Div", [tag + "_eps_once", tag + "_scale"], [tag + "_eps"], name=tag + "_eps2"),
        ])
        for node in consumers:
            for position, name in enumerate(node.input):
                if name == operand:
                    node.input[position] = tag + "_input"
        for position, name in enumerate(addition.input):
            if name == epsilon:
                addition.input[position] = tag + "_eps"
        changes.append(prefix)
    nodes = []
    for node in model.graph.node:
        nodes.extend(insertions.get(node.name, []))
        nodes.append(copy.deepcopy(node))
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    model.graph.initializer.append(numpy_helper.from_array(np.array(1, np.float32), "dynamic_one"))
    onnx.checker.check_model(model)
    onnx.save(model, str(args.output_dir / "temporal_layers_6_7.onnx"))
    manifest = json.loads((args.export_dir / "manifest.json").read_text())
    shard = next(item for item in manifest["shards"] if item["start_layer"] == 6)
    if shard["end_layer_exclusive"] != 8:
        raise ValueError("Expected the layers 6:8 shard")
    save_json(args.output_dir / "shard.json", shard)
    save_json(args.output_dir / "patch.json", {
        "norms": changes,
        "scale": "max(reduce_max(abs(hidden), axis=-1, keepdims=True), 1)",
        "epsilon": "epsilon / scale / scale",
        "validation": "pending; original FP32 ONNX is the reference",
    })
    print("Patched four real norms:", changes, flush=True)


def verify(args: argparse.Namespace, use_cloud: bool) -> None:
    root = args.output_dir
    shard = json.loads((root / "shard.json").read_text())
    model_path = root / shard["onnx"]
    target = None
    if use_cloud:
        import qai_hub as hub

        if not (root / "cpu_pass.json").exists():
            raise ValueError("Run cpu validation first")
        state_path = root / "compile.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        with model_path.open("rb") as source:
            digest = hashlib.sha256()
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if state.get("sha256", digest.hexdigest()) != digest.hexdigest():
            raise ValueError("Model changed; use a new output directory")
        state["sha256"] = digest.hexdigest()
        if "source_model_id" not in state:
            state["source_model_id"] = hub.upload_model(str(model_path)).model_id
            save_json(state_path, state)
        if "compile_job_id" not in state:
            job = hub.submit_compile_job(model=hub.get_model(state["source_model_id"]),
                device=hub.Device(name="Samsung Galaxy S26 (Family)", os="16"),
                options="--target_runtime qnn_context_binary", name="moshi-layers-6-8-dynamic-norm")
            state["compile_job_id"] = job.job_id
            save_json(state_path, state)
        job = hub.get_job(state["compile_job_id"])
        job.wait()
        target = job.get_target_model()
        if target is None:
            raise RuntimeError("Compilation produced no target")
        state["target_model_id"] = target.model_id
        save_json(state_path, state)
    original = ort.InferenceSession(str(args.export_dir / shard["onnx"]), providers=["CPUExecutionProvider"])
    patched = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"]) if not use_cloud else None
    caches = [np.zeros(tuple(item.shape), np.float32) for item in original.get_inputs()[2:]]
    reports = []
    for frame in range(2):
        with np.load(args.chain_dir / f"layers_4_6_frame_{frame}.npz") as archive:
            hidden = archive["output_hidden"].copy()
        inputs = dict(zip(shard["input_names"], [hidden, np.array([frame], np.int32), *caches], strict=True))
        expected = original.run(shard["output_names"], inputs)
        if target is not None:
            actual = cloud_step(target.model_id, inputs, shard["output_names"], root / f"cloud_frame_{frame}")
        else:
            actual = patched.run(shard["output_names"], inputs)
        report = {"frame": frame, "hidden": metrics(actual[0], expected[0]), "cache": {}}
        for name, received, reference, previous in zip(shard["output_names"][1:], actual[1:], expected[1:], caches, strict=True):
            if received.shape != reference.shape or not np.isfinite(received).all():
                raise ValueError(f"Invalid cache {name}")
            for lower, upper in ((0, frame), (frame + 1, received.shape[2])):
                np.testing.assert_array_equal(received[:, :, lower:upper], previous[:, :, lower:upper])
            report["cache"][name] = {**metrics(received[:, :, frame], reference[:, :, frame]),
                                     "written_nonzero": int(np.count_nonzero(received[:, :, frame]))}
        if not use_cloud:
            for received, reference in zip(actual, expected, strict=True):
                np.testing.assert_allclose(received, reference, rtol=1e-4, atol=1e-5)
        caches = actual[1:]
        print(json.dumps(report), flush=True)
        reports.append(report)
    save_json(root / ("cloud_metrics.json" if use_cloud else "cpu_pass.json"), reports)
    print("Same-input two-frame validation completed; untouched cache: PASS. Inspect numerical errors.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "cpu", "cloud"))
    parser.add_argument("--export-dir", type=Path, default=Path("/data2/user/moshi-work/moshi-temporal-shards-all"))
    parser.add_argument("--chain-dir", type=Path, default=Path("/data2/user/moshi-work/moshi-cloud-chain"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "prepare":
        prepare(args)
    else:
        verify(args, args.mode == "cloud")


if __name__ == "__main__":
    main()

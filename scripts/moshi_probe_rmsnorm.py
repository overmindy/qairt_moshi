"""Extract real layer-6 RMSNorm and compare original/scaled graphs on AI Hub."""

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


def save_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def compare(actual: np.ndarray, expected: np.ndarray) -> dict:
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError("Invalid output shape or non-finite values")
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    rmse = float(np.sqrt(np.mean(difference ** 2)))
    rms = float(np.sqrt(np.mean(expected.astype(np.float64) ** 2)))
    return {"max_abs": float(np.abs(difference).max()), "rmse": rmse,
            "relative_rms": rmse / max(rms, 1e-12),
            "nonzero": int(np.count_nonzero(actual)),
            "absmax": float(np.abs(actual).max())}


def prepare(args: argparse.Namespace) -> None:
    if any(args.output_dir.iterdir()):
        raise ValueError("Use an empty output directory; existing experiments are preserved")
    source = onnx.load(str(args.export_dir / "temporal_layers_6_7.onnx"))
    projection = next(node for node in source.graph.node if node.op_type == "MatMul")
    output_name = projection.input[0]
    required = {output_name}
    nodes = []
    for node in reversed(source.graph.node):
        if required.intersection(node.output):
            nodes.append(copy.deepcopy(node))
            required.update(node.input)
    nodes.reverse()
    initializers = [copy.deepcopy(value) for value in source.graph.initializer if value.name in required]
    inputs = [copy.deepcopy(value) for value in source.graph.input if value.name in required]
    if [value.name for value in inputs] != ["hidden"]:
        raise ValueError("First projection does not have a hidden-only normalization prefix")
    if sum(node.op_type == "ReduceMean" for node in nodes) != 1:
        raise ValueError("Expected a single RMSNorm reduction")
    if not any("norm1" in node.name for node in nodes):
        raise ValueError("Expected the first layer's norm1 before the first projection")
    output = copy.deepcopy(inputs[0])
    output.name = output_name
    graph = helper.make_graph(nodes, "real_layer_6_norm1", inputs, [output], initializers)
    original = helper.make_model(graph, opset_imports=list(source.opset_import))
    original.ir_version = source.ir_version
    del source
    constants = {value.name: numpy_helper.to_array(value) for value in initializers}
    for node in nodes:
        if node.op_type == "Constant":
            for attribute in node.attribute:
                if attribute.name == "value":
                    constants[node.output[0]] = numpy_helper.to_array(attribute.t)
    reductions = {name for node in nodes if node.op_type == "ReduceMean" for name in node.output}
    additions = [node for node in nodes if node.op_type == "Add" and reductions.intersection(node.input)]
    if len(additions) != 1:
        raise ValueError("Could not uniquely identify mean-square plus epsilon")
    epsilon_names = [name for name in additions[0].input if name in constants]
    if len(epsilon_names) != 1 or constants[epsilon_names[0]].size != 1:
        raise ValueError("Expected a scalar epsilon")
    epsilon_name = epsilon_names[0]
    epsilon = float(constants[epsilon_name].item())
    if not 0 < epsilon < 0.01:
        raise ValueError(f"Unexpected epsilon: {epsilon}")
    scaled = copy.deepcopy(original)
    for node in scaled.graph.node:
        for index, name in enumerate(node.input):
            if name == "hidden":
                node.input[index] = "probe_scaled_hidden"
            if node.name == additions[0].name and name == epsilon_name:
                node.input[index] = "probe_scaled_epsilon"
    scaled.graph.initializer.extend([
        numpy_helper.from_array(np.array(4.0, dtype=np.float32), "probe_scale"),
        numpy_helper.from_array(np.array(epsilon / 16, dtype=np.float32), "probe_scaled_epsilon"),
    ])
    scaled.graph.node.insert(0, helper.make_node(
        "Div", ["hidden", "probe_scale"], ["probe_scaled_hidden"], name="probe_input_scale"
    ))
    for variant, model in (("original", original), ("scaled", scaled)):
        onnx.checker.check_model(model)
        onnx.save(model, str(args.output_dir / f"{variant}.onnx"))
    sessions = {variant: ort.InferenceSession(str(args.output_dir / f"{variant}.onnx"),
                                             providers=["CPUExecutionProvider"])
                for variant in ("original", "scaled")}
    samples = {}
    for frame in range(2):
        with np.load(args.chain_dir / f"layers_4_6_frame_{frame}.npz") as archive:
            hidden = archive["output_hidden"].copy()
        reference = sessions["original"].run(None, {"hidden": hidden})[0]
        actual = sessions["scaled"].run(None, {"hidden": hidden})[0]
        np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=1e-6)
        samples[f"hidden_{frame}"] = hidden
        samples[f"reference_{frame}"] = reference
        with np.errstate(over="ignore"):
            squared = hidden.astype(np.float16) ** 2
            scaled_squared = (hidden / 4).astype(np.float16) ** 2
        print(f"frame={frame} CPU parity: {compare(actual, reference)}", flush=True)
        print(f"FP16 square inf: original={np.isinf(squared).sum()} scaled={np.isinf(scaled_squared).sum()}")
    np.savez(args.output_dir / "samples.npz", **samples)
    save_json(args.output_dir / "extraction.json", {
        "source": str(args.export_dir / "temporal_layers_6_7.onnx"),
        "projection": projection.name, "output": output_name,
        "epsilon": epsilon, "scale": 4, "nodes": [node.name for node in nodes],
    })
    for variant in sessions:
        print(f"{variant}.onnx: {(args.output_dir / (variant + '.onnx')).stat().st_size} bytes")


def cloud(args: argparse.Namespace) -> None:
    import qai_hub as hub

    root = args.output_dir
    state_path = root / "jobs.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    device = hub.Device(name="Samsung Galaxy S26 (Family)", os="16")
    options = "--target_runtime qnn_context_binary"
    results_path = root / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    with np.load(root / "samples.npz") as archive:
        samples = {name: archive[name].copy() for name in archive.files}
    for variant in args.variants:
        model_path = root / f"{variant}.onnx"
        signature = hashlib.sha256(model_path.read_bytes() + (root / "samples.npz").read_bytes()).hexdigest()
        record = state.setdefault(variant, {"signature": signature})
        if record["signature"] != signature:
            raise ValueError("Artifacts changed; use a new experiment directory")
        if "source_model_id" not in record:
            model = hub.upload_model(str(model_path))
            record["source_model_id"] = model.model_id
            save_json(state_path, state)
        if "compile_job_id" not in record:
            job = hub.submit_compile_job(model=hub.get_model(record["source_model_id"]),
                                         device=device, options=options, name=f"moshi-real-norm6-{variant}")
            record["compile_job_id"] = job.job_id
            save_json(state_path, state)
        compile_job = hub.get_job(record["compile_job_id"])
        compile_job.wait()
        target = compile_job.get_target_model()
        if target is None:
            raise RuntimeError(f"No target: {record['compile_job_id']}")
        record["target_model_id"] = target.model_id
        save_json(state_path, state)
        for frame in range(2):
            key = f"frame_{frame}_job_id"
            if key not in record:
                job = hub.submit_inference_job(
                    model=target, device=device, inputs={"hidden": [samples[f"hidden_{frame}"].astype(
                        np.float16 if variant == "fp32" else np.float32)]},
                    name=f"moshi-real-norm6-{variant}-frame-{frame}",
                )
                record[key] = job.job_id
                save_json(state_path, state)
            destination = root / f"{variant}_frame_{frame}.npy"
            if destination.exists():
                actual = np.load(destination)
            else:
                job = hub.get_job(record[key])
                job.wait()
                outputs = job.download_output_data()
                if outputs is None or len(outputs) != 1:
                    raise RuntimeError(f"Expected one output from {record[key]}")
                values = next(iter(outputs.values()))
                if len(values) != 1:
                    raise ValueError("Expected one output sample")
                actual = np.asarray(values[0])
                temporary = destination.with_suffix(".tmp.npy")
                np.save(temporary, actual)
                temporary.replace(destination)
            result = compare(actual, samples[f"reference_{frame}"])
            results[f"{variant}_frame_{frame}"] = result
            print(f"{variant} frame={frame}: {json.dumps(result)}", flush=True)
            save_json(root / "results.json", results)
    print("Finished real RMSNorm probe; this does not establish full-shard parity.")


def prepare_candidates(args: argparse.Namespace) -> None:
    if any(args.output_dir.iterdir()):
        raise ValueError("Use an empty output directory")
    original = onnx.load(str(args.baseline_dir / "original.onnx"))
    metadata = json.loads((args.baseline_dir / "extraction.json").read_text())
    weights = [numpy_helper.to_array(value) for value in original.graph.initializer
               if numpy_helper.to_array(value).size == 4096]
    if len(weights) != 1:
        raise ValueError("Expected exactly one real 4096-element RMSNorm weight")
    epsilon = metadata["epsilon"]
    for variant in ("gemma", "dynamic", "fp32"):
        nodes = []
        initializers = [numpy_helper.from_array(weights[0].astype(np.float32), "alpha"),
                        numpy_helper.from_array(np.array(epsilon, np.float32), "epsilon"),
                        numpy_helper.from_array(np.array(1.0, np.float32), "one")]
        operand = "hidden"
        epsilon_operand = "epsilon"
        if variant == "fp32":
            nodes.append(helper.make_node("Cast", [operand], ["hidden_float"], to=onnx.TensorProto.FLOAT))
            operand = "hidden_float"
        if variant == "dynamic":
            nodes.extend([
                helper.make_node("Abs", [operand], ["absolute"]),
                helper.make_node("ReduceMax", ["absolute"], ["maximum"], axes=[-1], keepdims=1),
                helper.make_node("Max", ["maximum", "one"], ["scale"]),
                helper.make_node("Div", [operand, "scale"], ["scaled_hidden"]),
                helper.make_node("Div", ["epsilon", "scale"], ["epsilon_once"]),
                helper.make_node("Div", ["epsilon_once", "scale"], ["scaled_epsilon"]),
            ])
            operand, epsilon_operand = "scaled_hidden", "scaled_epsilon"
        nodes.extend([
            helper.make_node("Mul", [operand, operand], ["squared"]),
            helper.make_node("ReduceMean", ["squared"], ["mean_squared"], axes=[-1], keepdims=1),
            helper.make_node("Add", ["mean_squared", epsilon_operand], ["variance"]),
            helper.make_node("Sqrt", ["variance"], ["denominator"]),
            helper.make_node("Div", [operand, "denominator"], ["normalized"]),
            helper.make_node("Mul", ["normalized", "alpha"], ["output"]),
        ])
        graph = helper.make_graph(nodes, f"norm6_{variant}", [helper.make_tensor_value_info(
            "hidden", onnx.TensorProto.FLOAT16 if variant == "fp32" else onnx.TensorProto.FLOAT,
            [1, 1, 4096])], [helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT,
                                                      [1, 1, 4096])], initializers)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = original.ir_version
        onnx.checker.check_model(model)
        onnx.save(model, str(args.output_dir / f"{variant}.onnx"))
    reference_session = ort.InferenceSession(str(args.baseline_dir / "original.onnx"),
                                             providers=["CPUExecutionProvider"])
    samples = {}
    with np.load(args.baseline_dir / "samples.npz") as archive:
        for frame in range(2):
            hidden = archive[f"hidden_{frame}"].astype(np.float16).astype(np.float32)
            if not np.isfinite(hidden).all():
                raise ValueError("Input is outside FP16 range")
            samples[f"hidden_{frame}"] = hidden
            samples[f"reference_{frame}"] = reference_session.run(None, {"hidden": hidden})[0]
    for variant in ("gemma", "dynamic", "fp32"):
        session = ort.InferenceSession(str(args.output_dir / f"{variant}.onnx"),
                                       providers=["CPUExecutionProvider"])
        for frame in range(2):
            hidden = samples[f"hidden_{frame}"].astype(np.float16 if variant == "fp32" else np.float32)
            actual = session.run(None, {"hidden": hidden})[0]
            reference = samples[f"reference_{frame}"]
            np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=1e-6)
            print(f"CPU {variant} frame={frame}: {compare(actual, reference)}")
        for label, hidden in (("zero", np.zeros((1, 1, 4096), np.float32)),
                              ("small", np.full((1, 1, 4096), 1e-3, np.float32))):
            hidden = hidden.astype(np.float16).astype(np.float32)
            reference = reference_session.run(None, {"hidden": hidden})[0]
            actual = session.run(None, {"hidden": hidden.astype(
                np.float16 if variant == "fp32" else np.float32)})[0]
            np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=1e-6)
            print(f"CPU {variant} {label}: PASS")
    np.savez(args.output_dir / "samples.npz", **samples)
    save_json(args.output_dir / "extraction.json", {
        **metadata, "variants": ["gemma", "dynamic", "fp32"],
        "input_policy": "All candidates use identical FP16-representable values",
        "fp32_policy": "FP16 input, explicit Cast to FLOAT; backend precision is not guaranteed",
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "prepare-candidates", "cloud"))
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--variants", nargs="+", choices=("original", "scaled", "gemma", "dynamic", "fp32"),
                        default=["original", "scaled"])
    parser.add_argument("--export-dir", type=Path,
                        default=Path("/data2/user/moshi-work/moshi-temporal-shards-all"))
    parser.add_argument("--chain-dir", type=Path,
                        default=Path("/data2/user/moshi-work/moshi-cloud-chain"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "prepare":
        prepare(args)
    elif args.mode == "prepare-candidates":
        if args.baseline_dir is None:
            parser.error("prepare-candidates requires --baseline-dir")
        prepare_candidates(args)
    else:
        cloud(args)


if __name__ == "__main__":
    main()

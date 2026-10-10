"""Isolated first-RMSNorm overflow experiment; reuse every existing QDQ encoding."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper
import onnxruntime as ort

from moshi_verify_dynamic_shard import patch_dynamic_rmsnorms
from moshi_probe_quantized_graph_cloud import _probe_metrics
from moshi_verify_lm_graph_set_cloud import _sha256, _write_json


def remove_norm_activation_qdq(model):
    """Remove only QDQ pairs fed by named norm tensors; retain weight QDQ."""
    prefixes = ("/norm1/", "/norm2/", "/norm1_1/", "/norm2_1/")
    quantizers = [n for n in model.graph.node if n.op_type == "QuantizeLinear"
                  and n.input[0].startswith(prefixes)]
    rewrites, removed = {}, set()
    for quantizer in quantizers:
        consumers = [n for n in model.graph.node if quantizer.output[0] in n.input]
        if not consumers or any(n.op_type != "DequantizeLinear" or n.input[0] != quantizer.output[0] for n in consumers):
            raise ValueError("Norm quantizer has non-DQ consumers; refusing unsafe removal")
        removed.add(quantizer.name)
        for dq in consumers:
            if list(dq.input[1:]) != list(quantizer.input[1:]):
                raise ValueError("Norm Q/DQ encodings differ")
            rewrites[dq.output[0]] = quantizer.input[0]
            removed.add(dq.name)
    def original(value):
        seen = set()
        while value in rewrites:
            if value in seen:
                raise ValueError("Cyclic norm rewrite")
            seen.add(value)
            value = rewrites[value]
        return value
    nodes = []
    for node in model.graph.node:
        if node.name in removed:
            continue
        for i, value in enumerate(node.input):
            node.input[i] = original(value)
        nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    for output in model.graph.output:
        if output.name in rewrites:
            raise ValueError("Graph output alias requires a separate explicit mapping")
    return sorted(removed)


def stabilize_denominator(model, prefix):
    """Rescale only the square, then restore RMS units BEFORE existing QDQ.

    Unlike rescaling the complete norm, this leaves reciprocal/gamma/numerator
    and every existing quantizer input range in their original units.
    """
    producers = {v: n for n in model.graph.node for v in n.output}
    reductions = [n for n in model.graph.node if n.name.startswith(prefix) and n.op_type == "ReduceMean"]
    if len(reductions) != 1:
        raise ValueError("Expected one denominator reduction")
    reduction = reductions[0]
    square = producers[reduction.input[0]]
    if square.op_type != "Pow":
        raise ValueError("Expected unquantized square before reduction")
    constants = {v.name: numpy_helper.to_array(v) for v in model.graph.initializer}
    for n in model.graph.node:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    constants[n.output[0]] = numpy_helper.to_array(a.t)
    if square.input[1] not in constants or float(constants[square.input[1]].item()) != 2:
        raise ValueError("Expected square exponent two")
    additions = [n for n in model.graph.node if n.op_type == "Add" and reduction.output[0] in n.input]
    if len(additions) != 1:
        raise ValueError("Quantized reduction is not supported by this isolated experiment")
    addition = additions[0]
    epsilon = next(v for v in addition.input if v != reduction.output[0])
    if epsilon not in constants or not 0 < float(constants[epsilon].item()) < 0.01:
        raise ValueError("Expected positive scalar epsilon")
    roots = [n for n in model.graph.node if n.op_type == "Sqrt" and addition.output[0] in n.input]
    if len(roots) != 1:
        raise ValueError("Expected unquantized square root")
    sqrt = roots[0]
    tag = "stable_denominator_" + str(sum(v.name.startswith("stable_denominator_") and v.name.endswith("_one") for v in model.graph.initializer))
    model.graph.initializer.append(numpy_helper.from_array(np.array(1, np.float32), tag + "_one"))
    opset = next(v.version for v in model.opset_import if v.domain in ("", "ai.onnx"))
    if opset >= 18:
        model.graph.initializer.append(numpy_helper.from_array(np.array([-1], np.int64), tag + "_axes"))
        maximum = helper.make_node("ReduceMax", [tag + "_abs", tag + "_axes"], [tag + "_max"], keepdims=1)
    else:
        maximum = helper.make_node("ReduceMax", [tag + "_abs"], [tag + "_max"], axes=[-1], keepdims=1)
    before = [helper.make_node("Abs", [square.input[0]], [tag + "_abs"]), maximum,
              helper.make_node("Max", [tag + "_max", tag + "_one"], [tag + "_scale"]),
              helper.make_node("Div", [square.input[0], tag + "_scale"], [tag + "_input"])]
    square.input[0] = tag + "_input"
    epsilon_nodes = [helper.make_node("Div", [epsilon, tag + "_scale"], [tag + "_eps1"]),
                     helper.make_node("Div", [tag + "_eps1", tag + "_scale"], [tag + "_eps2"])]
    for i, value in enumerate(addition.input):
        if value == epsilon:
            addition.input[i] = tag + "_eps2"
    original_output = sqrt.output[0]
    sqrt.output[0] = tag + "_rms_scaled"
    restore = helper.make_node("Mul", [tag + "_rms_scaled", tag + "_scale"], [original_output])
    for i, node in enumerate(before + epsilon_nodes + [restore]):
        node.name = tag + "_op_" + str(i)
    nodes = []
    for node in model.graph.node:
        if node.name == square.name:
            nodes.extend(before)
        if node.name == addition.name:
            nodes.extend(epsilon_nodes)
        nodes.append(node)
        if node.name == sqrt.name:
            nodes.append(restore)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    onnx.checker.check_model(model)
    return prefix


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-onnx", type=Path, required=True)
    p.add_argument("--calibration-dir", type=Path, required=True)
    p.add_argument("--graph", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--stable-denominator", action="append", default=[],
                   help="Norm prefix whose RMS is restored to original units before its existing QDQ")
    p.add_argument("--float-norm-activations", action="store_true")
    p.add_argument("--reference-onnx", type=Path,
                   help="Original FP32 reference required for intentional norm precision refinement")
    args = p.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Use a fresh output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = onnx.load(str(args.source_onnx))
    # This targeted experiment refuses quantized intermediates inside the
    # selected norm; its old encodings would become invalid after rescaling.
    selected = [n for n in model.graph.node if n.name.startswith("/norm1/")]
    if any(n.op_type in ("QuantizeLinear", "DequantizeLinear") for n in selected):
        raise ValueError("First norm contains QDQ; cannot safely rescale its encoded intermediates")
    initializer_hashes = {v.name: hashlib.sha256(numpy_helper.to_array(v).tobytes()).hexdigest()
                          for v in model.graph.initializer}
    names = [v.name for v in model.graph.output]
    weight_qdq = {n.name: n.SerializeToString() for n in model.graph.node
                  if n.op_type in ("QuantizeLinear", "DequantizeLinear")
                  and any(value.startswith("onnx::MatMul_") for value in n.input)}
    removed_qdq = remove_norm_activation_qdq(model) if args.float_norm_activations else []
    if args.float_norm_activations and not args.reference_onnx:
        raise ValueError("Norm precision refinement requires the original FP32 reference")
    changes = patch_dynamic_rmsnorms(model, None if args.float_norm_activations else ["/norm1/ReduceMean"])
    denominator_changes = [stabilize_denominator(model, prefix) for prefix in args.stable_denominator]
    if not all(hashlib.sha256(numpy_helper.to_array(v).tobytes()).hexdigest() == initializer_hashes[v.name]
               for v in model.graph.initializer if v.name in initializer_hashes):
        raise ValueError("Existing QDQ initializer changed")
    if any(next((n.SerializeToString() for n in model.graph.node if n.name == name), None) != data
           for name, data in weight_qdq.items()):
        raise ValueError("Projection weight QDQ changed")
    output = args.output_dir / "model.onnx"
    onnx.save_model(model, str(output), save_as_external_data=True,
                    all_tensors_to_one_file=True, location="model.data", size_threshold=1024)
    del model
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    reference = args.reference_onnx if args.float_norm_activations else args.source_onnx
    original = ort.InferenceSession(str(reference), options, providers=["CPUExecutionProvider"])
    patched = ort.InferenceSession(str(output), options, providers=["CPUExecutionProvider"])
    calibration = json.loads((args.calibration_dir / "manifest.json").read_text())
    rows = []
    for frame in (0, 1):
        sample = next(s for s in calibration["graphs"][args.graph]["samples"]
                      if s["source_id"] == "clean-000" and s["frame"] == frame)
        with np.load(args.calibration_dir / sample["file"]) as a:
            feed = {v.name: a[v.name] for v in original.get_inputs()}
        expected = original.run(None, feed)
        actual = patched.run(names, feed)
        metrics = {name: _probe_metrics(a, e) for name, a, e in zip(names, actual, expected)}
        limit = 0.02 if args.float_norm_activations else 1e-4
        passed = all(v["finite"] and v["relative_rms"] <
                     (0.05 if args.float_norm_activations and index > 0 else limit)
                     for index, v in enumerate(metrics.values()))
        rows.append({"frame": frame, "outputs": metrics, "passed": passed})
        print(json.dumps(rows[-1]), flush=True)
    _write_json(args.output_dir / "patch.json", {
        "graph": args.graph, "source_sha256": _sha256(args.source_onnx),
        "patched_sha256": _sha256(output), "norms": changes,
        "existing_initializer_values_unchanged": True,
        "stable_denominators": denominator_changes,
        "precision_refinement": args.float_norm_activations,
        "removed_norm_qdq_nodes": removed_qdq,
        "projection_weight_qdq_unchanged": True,
        "cpu_reference_sha256": _sha256(reference),
        "cpu_relative_rms_limit": limit,
        "cpu_cache_relative_rms_limit": 0.05 if args.float_norm_activations else limit,
        "samples": rows, "passed": all(r["passed"] for r in rows),
    })
    if not all(r["passed"] for r in rows):
        raise RuntimeError("CPU reference gate failed; do not convert")


if __name__ == "__main__":
    main()

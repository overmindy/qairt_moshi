"""Locate remaining quantization error before paying for another cloud probe."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import onnx
import onnxruntime as ort

from moshi_probe_quantized_graph_cloud import _probe_metrics
from moshi_verify_lm_graph_set_cloud import _sha256, _write_json


def after_qdq(model, tensor):
    quantizers = [n for n in model.graph.node if n.op_type == "QuantizeLinear" and n.input[0] == tensor]
    if not quantizers:
        return tensor
    if len(quantizers) != 1:
        raise ValueError("Ambiguous activation quantizer")
    dequantizers = [n for n in model.graph.node if n.op_type == "DequantizeLinear" and n.input[0] == quantizers[0].output[0]]
    if len(dequantizers) != 1:
        raise ValueError("Ambiguous activation dequantizer")
    return dequantizers[0].output[0]


def inferred_values(model):
    """Preserve actual intermediate dtypes, including integer position arithmetic."""
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=False)
    values = {v.name: v for v in (*inferred.graph.input, *inferred.graph.value_info,
                                 *inferred.graph.output)}
    return values


def floating_value(values, name):
    if name not in values or not values[name].type.HasField("tensor_type"):
        raise ValueError(f"Cannot infer diagnostic output type: {name}")
    return values[name].type.tensor_type.elem_type in (
        onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16, onnx.TensorProto.DOUBLE,
        onnx.TensorProto.BFLOAT16,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-onnx", type=Path, required=True)
    p.add_argument("--source-onnx", type=Path, required=True)
    p.add_argument("--calibration-dir", type=Path, required=True)
    p.add_argument("--graph", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "report.json").exists():
        raise ValueError("Completed diagnostic exists; inspect it instead of rerunning")
    reference = onnx.load(str(args.reference_onnx))
    source = onnx.load(str(args.source_onnx))
    ref_values = inferred_values(reference)
    source_values = inferred_values(source)
    nodes = {n.name: n for n in source.graph.node}
    pairs = []
    skipped = []
    for node in reference.graph.node:
        selected = (node.op_type == "MatMul" or (node.op_type == "Add" and node.name.startswith("/Add"))
                    or ("norm" in node.name and node.name.endswith("/Cast_1")))
        if not selected:
            continue
        if len(node.output) != 1:
            raise ValueError(f"Stage has multiple outputs: {node.name}")
        if not floating_value(ref_values, node.output[0]):
            skipped.append(node.name)
            continue
        peer = nodes.get(node.name)
        if peer is None or peer.op_type != node.op_type or len(peer.output) != 1 or len(node.output) != 1:
            raise ValueError(f"Stage mapping changed: {node.name}")
        pairs.append({"node": node.name, "op": node.op_type, "reference": node.output[0],
                      "before_qdq": peer.output[0], "after_qdq": after_qdq(source, peer.output[0])})
    if not pairs:
        raise ValueError("No comparable stages")
    ref_outputs = list(dict.fromkeys(v["reference"] for v in pairs))
    qdq_outputs = list(dict.fromkeys(v[k] for v in pairs for k in ("before_qdq", "after_qdq")))
    def session(model, names, values):
        del model.graph.output[:]
        for name in names:
            if not floating_value(values, name):
                raise ValueError(f"Floating stage maps to non-floating output: {name}")
            model.graph.output.add().CopyFrom(values[name])
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.inter_op_num_threads = 1
        # Fully loaded weights stay in memory; do not duplicate model files.
        return ort.InferenceSession(model.SerializeToString(), opts, providers=["CPUExecutionProvider"])
    ref_session = session(reference, ref_outputs, ref_values)
    del reference
    qdq_session = session(source, qdq_outputs, source_values)
    del source
    calibration = json.loads((args.calibration_dir / "manifest.json").read_text())
    result = {"graph": args.graph, "reference_sha256": _sha256(args.reference_onnx),
              "source_sha256": _sha256(args.source_onnx), "diagnostic_only": True,
              "skipped_nonfloating_nodes": skipped, "samples": []}
    for frame in (0, 1):
        sample = next(s for s in calibration["graphs"][args.graph]["samples"]
                      if s["source_id"] == "clean-000" and s["frame"] == frame)
        with np.load(args.calibration_dir / sample["file"]) as arrays:
            feed = {v.name: arrays[v.name] for v in ref_session.get_inputs()}
        expected = dict(zip(ref_outputs, ref_session.run(ref_outputs, feed)))
        actual = dict(zip(qdq_outputs, qdq_session.run(qdq_outputs, feed)))
        rows = []
        for pair in pairs:
            target = expected[pair["reference"]]
            before, after = actual[pair["before_qdq"]], actual[pair["after_qdq"]]
            if before.shape != target.shape or after.shape != target.shape:
                raise ValueError(f"Stage shape mismatch: {pair['node']}")
            rows.append({"node": pair["node"], "op": pair["op"], "shape": list(target.shape),
                         "reference_absmax": float(np.max(np.abs(target))),
                         "before_qdq": _probe_metrics(before, target),
                         "after_qdq": _probe_metrics(after, target),
                         "local_qdq_change": _probe_metrics(after, before)})
        result["samples"].append({"frame": frame, "stages": rows})
        _write_json(args.output_dir / "report.json", result)
        print(json.dumps(result["samples"][-1]), flush=True)


if __name__ == "__main__":
    main()

"""Isolated first-RMSNorm overflow experiment; reuse every existing QDQ encoding."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort

from moshi_verify_dynamic_shard import patch_dynamic_rmsnorms
from moshi_probe_quantized_graph_cloud import _probe_metrics
from moshi_verify_lm_graph_set_cloud import _sha256, _write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-onnx", type=Path, required=True)
    p.add_argument("--calibration-dir", type=Path, required=True)
    p.add_argument("--graph", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
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
    changes = patch_dynamic_rmsnorms(model, ["/norm1/ReduceMean"])
    if not all(hashlib.sha256(numpy_helper.to_array(v).tobytes()).hexdigest() == initializer_hashes[v.name]
               for v in model.graph.initializer if v.name in initializer_hashes):
        raise ValueError("Existing QDQ initializer changed")
    output = args.output_dir / "model.onnx"
    onnx.save_model(model, str(output), save_as_external_data=True,
                    all_tensors_to_one_file=True, location="model.data", size_threshold=1024)
    del model
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    original = ort.InferenceSession(str(args.source_onnx), options, providers=["CPUExecutionProvider"])
    patched = ort.InferenceSession(str(output), options, providers=["CPUExecutionProvider"])
    calibration = json.loads((args.calibration_dir / "manifest.json").read_text())
    rows = []
    for frame in (0, 1):
        sample = next(s for s in calibration["graphs"][args.graph]["samples"]
                      if s["source_id"] == "clean-000" and s["frame"] == frame)
        with np.load(args.calibration_dir / sample["file"]) as a:
            feed = {v.name: a[v.name] for v in original.get_inputs()}
        expected = original.run(names, feed)
        actual = patched.run(names, feed)
        metrics = {name: _probe_metrics(a, e) for name, a, e in zip(names, actual, expected)}
        passed = all(v["finite"] and v["relative_rms"] < 1e-4 for v in metrics.values())
        rows.append({"frame": frame, "outputs": metrics, "passed": passed})
        print(json.dumps(rows[-1]), flush=True)
    _write_json(args.output_dir / "patch.json", {
        "graph": args.graph, "source_sha256": _sha256(args.source_onnx),
        "patched_sha256": _sha256(output), "norms": changes,
        "existing_initializer_values_unchanged": True,
        "samples": rows, "passed": all(r["passed"] for r in rows),
    })
    if not all(r["passed"] for r in rows):
        raise RuntimeError("Patch changed existing QDQ CPU results; do not convert")


if __name__ == "__main__":
    main()

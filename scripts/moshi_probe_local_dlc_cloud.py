"""Validate a locally converted Temporal DLC without re-quantizing its QDQ.

Upload once, reuse matching inference receipts, compare two captured frames
and a recurrent frame-1 call against the original FP32 ONNX. Publish a separate
candidate manifest only after numerical (not just finite-output) gates pass.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import qai_hub as hub

from moshi_probe_quantized_graph_cloud import _graph_specs, _probe_metrics
from moshi_verify_lm_graph_set_cloud import _cloud_batch, _sha256, _write_json


def _metadata(dlc: Path) -> dict:
    from qti.aisw.dlc_utils import snpe_dlc_utils
    reader = snpe_dlc_utils.modeltools.IrDlcReader()
    reader.open(str(dlc))
    graph = reader.get_ir_graph()
    inputs = [{"name": tensor.name(), "shape": list(tensor.dims()), "dtype": int(tensor.data_type()), "id": int(tensor.id())}
              for tensor in graph.get_input_tensors_to_graph()]
    # QNN application input tensors are enumerated by serialized tensor ID.
    inputs.sort(key=lambda item: item["id"])
    outputs = {}
    for op in graph.get_ops():
        for tensor in op.outputs():
            if tensor.is_app_read_tensor():
                outputs[tensor.name()] = {"shape": list(tensor.dims()), "dtype": int(tensor.data_type())}
    return {"inputs": inputs, "outputs": outputs}


def _evaluate(actual, expected, feed, spec, sample, args) -> dict:
    outputs = {}
    passed = True
    for index, (name, value, reference) in enumerate(zip(spec["output_names"], actual, expected, strict=True)):
        if value.shape != reference.shape:
            raise ValueError(f"Output shape changed: {name}")
        entry = _probe_metrics(value, reference)
        if not entry["finite"]:
            passed = False
        elif name == "audio_token":
            entry["token_match"] = bool(np.array_equal(value, reference))
            passed &= entry["token_match"]
        elif name.startswith("output_layer_"):
            slot = (int(np.asarray(feed["position"]).reshape(-1)[0])
                    if "position" in feed else int(spec["codebook"]))
            if not 0 <= slot < value.shape[2]:
                raise ValueError("Probe only supports in-range cache positions")
            written = _probe_metrics(value[:, :, slot], reference[:, :, slot])
            previous = feed[name.removeprefix("output_")]
            preserved = max((float(np.max(np.abs(value[:, :, start:end] - previous[:, :, start:end])))
                             for start, end in ((0, slot), (slot + 1, value.shape[2])) if start < end), default=0.0)
            entry.update({"written": written, "preserved_max_abs": preserved,
                          "written_nonzero": int(np.count_nonzero(value[:, :, slot]))})
            passed &= written["finite"] and written["relative_rms"] <= args.cache_relative_rms_limit
            passed &= preserved <= args.preserved_absolute_limit
        else:
            passed &= entry["relative_rms"] <= args.hidden_relative_rms_limit
        outputs[name] = entry
    return {"source_id": sample["source_id"], "frame": sample["frame"], "outputs": outputs,
            "passed": bool(passed)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dlc", type=Path, required=True)
    parser.add_argument("--conversion-receipt", type=Path, required=True)
    parser.add_argument("--onnx-dir", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--graph", default="temporal_layers_0_1")
    parser.add_argument("--source-id", default="clean-000")
    parser.add_argument("--hidden-relative-rms-limit", type=float, default=0.02)
    parser.add_argument("--cache-relative-rms-limit", type=float, default=0.05)
    parser.add_argument("--preserved-absolute-limit", type=float, default=0.001)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--source-onnx", type=Path, help="Downloaded QDQ used for DLC conversion; verifies output names")
    args = parser.parse_args()
    for limit in (args.hidden_relative_rms_limit, args.cache_relative_rms_limit, args.preserved_absolute_limit):
        if not np.isfinite(limit) or limit < 0:
            raise ValueError("Numerical limits must be finite and non-negative")
    conversion = json.loads(args.conversion_receipt.read_text())
    if conversion.get("status") != "success" or conversion.get("format") != "moshi-qairt-rmsnorm-guard-v1":
        raise ValueError("Successful guarded conversion receipt is required")
    argv = conversion["converter_args"]
    if "--preserve_onnx_output_order" not in argv or Path(argv[argv.index("--output_path") + 1]).resolve() != args.dlc.resolve():
        raise ValueError("DLC/order does not match the conversion receipt")
    manifest_path = args.onnx_dir / "manifest.json"
    calibration_path = args.calibration_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    calibration = json.loads(calibration_path.read_text())
    baseline = json.loads(args.baseline_manifest.read_text())
    graph_hash = _sha256(manifest_path)
    calibration_hash = _sha256(calibration_path)
    if calibration.get("graph_manifest_sha256") != graph_hash or baseline.get("graph_manifest_sha256") != graph_hash:
        raise ValueError("ONNX/calibration/baseline manifests do not match")
    if baseline.get("calibration_manifest_sha256") != calibration_hash:
        raise ValueError("Baseline/calibration manifests do not match")
    spec = _graph_specs(manifest)[args.graph]
    meta = _metadata(args.dlc)
    input_order = [item["name"] for item in meta["inputs"]]
    if args.source_onnx:
        import onnx
        source_graph = onnx.load(str(args.source_onnx), load_external_data=False).graph
        output_names = [output.name for output in source_graph.output]
        if len(output_names) != len(spec["output_names"]):
            raise ValueError("Source output count changed")
        # Downloads may rename to output_N; allow only known positional names
        # or original semantic names, never arbitrary guessed permutations.
        if output_names not in (spec["output_names"], [f"output_{i}" for i in range(len(output_names))]):
            raise ValueError("Unrecognized source output mapping")
    else:
        output_names = [f"output_{index}" for index in range(len(spec["output_names"]))]
    if set(input_order) != set(spec["input_names"]) or set(meta["outputs"]) != set(output_names):
        raise ValueError("Local DLC interface does not match the captured graph")
    # HTP input ABI is int32, whereas the original ONNX/calibration uses int64.
    for item in meta["inputs"]:
        if item["name"] in ("position", "sequence", "previous_token") and item["dtype"] != 0x0032:
            raise ValueError(f"DLC integer input must be int32: {item['name']}")
    samples = [next(s for s in calibration["graphs"][args.graph]["samples"]
                    if s["source_id"] == args.source_id and s["frame"] == frame) for frame in (0, 1)]
    feeds = []
    for sample in samples:
        with np.load(args.calibration_dir / sample["file"]) as arrays:
            feeds.append({name: arrays[name].copy() for name in spec["input_names"]})
    for item in meta["inputs"]:
        if list(feeds[0][item["name"]].shape) != item["shape"]:
            raise ValueError(f"DLC input shape changed: {item['name']}")
    device = baseline["target_device"]
    config = {"dlc_sha256": _sha256(args.dlc), "conversion_receipt_sha256": _sha256(args.conversion_receipt),
              "graph_manifest_sha256": graph_hash, "calibration_manifest_sha256": calibration_hash,
              "baseline_manifest_sha256": _sha256(args.baseline_manifest), "graph": args.graph,
              "target_device": device, "input_order": input_order, "output_names": output_names}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir / "jobs.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"config": config}
    if state["config"] != config:
        old_without_order = {k: v for k, v in state["config"].items() if k != "input_order"}
        new_without_order = {k: v for k, v in config.items() if k != "input_order"}
        if old_without_order != new_without_order or not args.retry_failed:
            raise ValueError("DLC/configuration changed; use a new probe directory")
        state["config"] = config
        _write_json(state_path, state)
    _write_json(args.output_dir / "dlc_metadata.json", meta)
    if not state.get("model_id"):
        state["model_id"] = hub.upload_model(str(args.dlc), name=f"moshi-{args.graph}-guarded-qdq").model_id
        _write_json(state_path, state)
    actual = _cloud_batch(model_id=state["model_id"], device=device, samples=feeds,
                          input_order=input_order, output_names=output_names,
                          stem=args.output_dir / "captured_frames_0_1", retry_failed=args.retry_failed)
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.onnx_dir / spec["onnx"]), options, providers=["CPUExecutionProvider"])
    expected = [session.run(spec["output_names"], feed) for feed in feeds]
    rows = [_evaluate(a, e, f, spec, s, args) for a, e, f, s in zip(actual, expected, feeds, samples, strict=True)]
    report = {"format": "moshi-local-guarded-dlc-probe-v1", "model_id": state["model_id"],
              "graph": args.graph, "samples": rows, "recurrent": [], "passed": False,
              "limits": {"hidden_relative_rms": args.hidden_relative_rms_limit,
                         "written_cache_relative_rms": args.cache_relative_rms_limit,
                         "preserved_absolute": args.preserved_absolute_limit}}
    _write_json(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    if not all(row["passed"] for row in rows):
        raise RuntimeError("Captured-frame numerical gate failed; candidate not published")
    validation_stem = "captured_frames_0_1"
    if "position" in feeds[1]:
        recurrent = {name: value.copy() for name, value in feeds[1].items()}
        for name, cache in zip(spec["input_names"][2:], actual[0][1:], strict=True):
            recurrent[name] = cache.copy()
        recurrent_actual = _cloud_batch(model_id=state["model_id"], device=device, samples=[recurrent],
                                        input_order=input_order, output_names=output_names,
                                        stem=args.output_dir / "recurrent_frame_1", retry_failed=args.retry_failed)[0]
        recurrent_expected = session.run(spec["output_names"], recurrent)
        report["recurrent"] = [_evaluate(recurrent_actual, recurrent_expected, recurrent, spec, samples[1], args)]
        report["passed"] = report["recurrent"][0]["passed"]
        validation_stem = "recurrent_frame_1"
    else:
        report["passed"] = True
    _write_json(args.output_dir / "report.json", report)
    if not report["passed"]:
        raise RuntimeError("Recurrent numerical gate failed; candidate not published")
    candidate = copy.deepcopy(baseline)
    record = candidate["graphs"][args.graph]
    old_model = record["compiled_model_id"]
    record.update({"compiled_model_id": state["model_id"], "compile_job_id": None,
                   "compiled_input_order": input_order,
                   "local_compile": {"kind": "qairt_rmsnorm_guard", "model_id": state["model_id"],
                                     "dlc_sha256": config["dlc_sha256"], "guard": conversion["guard"],
                                     "numerical_probe_passed": True,
                                     "validated_inference_job_id": json.loads((args.output_dir / f"{validation_stem}.json").read_text())["job_id"],
                                     "replaces_model_id": old_model}})
    destination = args.output_dir / "candidate/quantization_manifest.json"
    destination.parent.mkdir(exist_ok=True)
    _write_json(destination, candidate)
    print(f"Guarded DLC captured + recurrent numerical PASS; candidate: {destination}", flush=True)


if __name__ == "__main__":
    main()

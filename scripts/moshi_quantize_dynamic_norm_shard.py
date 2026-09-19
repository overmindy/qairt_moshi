"""Quantize one CPU-validated dynamic-RMSNorm shard and build a gated LM candidate.

Every other graph is copied from an existing compiled quantization manifest.
This is a direct two-frame probe, not an end-to-end streaming validation.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import qai_hub as hub

from moshi_probe_dynamic_norm_cloud import FORMAT as PATCH_FORMAT, _sample_feed
from moshi_probe_float_compile_cloud import _source_and_sample
from moshi_probe_quantized_graph_cloud import _probe_metrics, _session
from moshi_quantize_lm_graph_set import (
    COMPILE_RUNTIME,
    Precision,
    _compile_options,
    _input_specs,
    _load_calibration_entries,
    _quantize_options,
)
from moshi_verify_lm_graph_set_cloud import _cloud_batch, _sha256, _write_json


FORMAT = "moshi-lm-dynamic-rmsnorm-quantization-v1"
MAX_RELATIVE_RMS = 0.05
MAX_PRESERVED_ABS = 0.001


def _require_target(job: Any, stage: str) -> str:
    job.wait()
    status = job.get_status()
    if not status.success:
        raise RuntimeError(f"{stage} failed ({status}): {job.url}")
    model = job.get_target_model()
    if model is None:
        raise RuntimeError(f"{stage} returned no target model: {job.url}")
    return model.model_id


def _patch_and_baseline(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path]:
    config, spec, _ = _source_and_sample(args)
    baseline_path = args.source_quantization_manifest
    baseline = json.loads(baseline_path.read_text())
    if baseline.get("graph_manifest_sha256") != config["graph_manifest_sha256"]:
        raise ValueError("Baseline graph manifest changed")
    if baseline.get("calibration_manifest_sha256") != config["calibration_manifest_sha256"]:
        raise ValueError("Baseline calibration manifest changed")
    if baseline.get("target_device") != config["target_device"]:
        raise ValueError("Baseline target device differs from float probe")
    if baseline.get("compile_options") != _compile_options():
        raise ValueError("Baseline compile options changed")
    if any(not item.get("compiled_model_id") for item in baseline["graphs"].values()):
        raise ValueError("Baseline must have compiled IDs for every graph")

    patch_path = args.patch_dir / f"{args.graph}.onnx"
    receipt = json.loads((args.patch_dir / "cpu_pass.json").read_text())
    if (
        receipt.get("format") != PATCH_FORMAT
        or receipt.get("graph") != args.graph
        or receipt.get("graph_manifest_sha256") != config["graph_manifest_sha256"]
        or receipt.get("calibration_manifest_sha256")
        != config["calibration_manifest_sha256"]
        or receipt.get("source_onnx_sha256") != _sha256(args.onnx_dir / spec["onnx"])
        or receipt.get("patched_onnx_sha256") != _sha256(patch_path)
        or len(receipt.get("cpu_samples", [])) != 2
    ):
        raise ValueError("Patched ONNX lacks matching two-frame CPU parity receipt")
    return config, spec, baseline, patch_path


def _job_state(args: argparse.Namespace, plan: dict[str, Any]) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir / "jobs.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"plan": plan}
    if state.get("plan") != plan:
        raise ValueError("Quantization plan changed; use a new output directory")
    _write_json(state_path, state)
    return state


def _save_state(args: argparse.Namespace, state: dict[str, Any]) -> None:
    _write_json(args.output_dir / "jobs.json", state)


def _retry_failed_job(
    args: argparse.Namespace, state: dict[str, Any], job_key: str, model_key: str
) -> None:
    job_id = state.get(job_key)
    if not job_id:
        return
    job = hub.get_job(job_id)
    if not job.get_status().failure:
        return
    if not args.retry_failed:
        raise RuntimeError(
            f"Recorded {job_key} failed: {job.url}; pass --retry-failed "
            "to submit a replacement while preserving its ID"
        )
    state.setdefault("failed_job_ids", []).append(job_id)
    state.pop(job_key, None)
    state.pop(model_key, None)
    _save_state(args, state)


def _source_model_id(
    args: argparse.Namespace, state: dict[str, Any], patch_path: Path, patch_sha: str
) -> str:
    if state.get("source_model_id"):
        return state["source_model_id"]
    upload_path = args.patch_dir / "upload.json"
    upload = json.loads(upload_path.read_text()) if upload_path.exists() else {}
    if upload and upload.get("patched_onnx_sha256") != patch_sha:
        raise ValueError("Float-probe upload does not match patched ONNX")
    model_id = upload.get("source_model_id")
    if not model_id:
        model_id = hub.upload_model(str(patch_path)).model_id
    state["source_model_id"] = model_id
    _save_state(args, state)
    print(f"Patched source model: {model_id}", flush=True)
    return model_id


def _quantize_and_compile(
    args: argparse.Namespace,
    state: dict[str, Any],
    calibration_graph: dict[str, Any],
    device: dict[str, str],
    options: str,
) -> tuple[str, Any]:
    _retry_failed_job(args, state, "quantize_job_id", "quantized_model_id")
    if "quantize_job_id" not in state:
        entries = _load_calibration_entries(args.calibration_dir, calibration_graph)
        precision = Precision.w8a16_mixed_fp16
        job = hub.submit_quantize_job(
            model=hub.get_model(state["source_model_id"]),
            calibration_data=entries,
            activations_dtype=precision.activations_type,
            weights_dtype=precision.weights_type,
            name=f"moshi-{args.graph}-dynamic-norm-mixed-fp16-100",
            options=options,
        )
        state["quantize_job_id"] = job.job_id
        _save_state(args, state)
        print(f"Quantize submitted: {job.url}", flush=True)
        del entries
    else:
        job = hub.get_job(state["quantize_job_id"])
        print(f"Quantize resumed: {job.url}", flush=True)
    quantized_model_id = _require_target(job, "Quantize")
    if state.get("quantized_model_id", quantized_model_id) != quantized_model_id:
        raise ValueError("Recorded quantized model ID changed")
    state["quantized_model_id"] = quantized_model_id
    _save_state(args, state)

    _retry_failed_job(args, state, "compile_job_id", "compiled_model_id")
    if "compile_job_id" not in state:
        job = hub.submit_compile_job(
            model=hub.get_model(quantized_model_id),
            input_specs=_input_specs(args.calibration_dir, calibration_graph),
            device=hub.Device(**device),
            name=f"moshi-{args.graph}-dynamic-norm-mixed-fp16-100-dlc",
            options=_compile_options(),
        )
        state["compile_job_id"] = job.job_id
        _save_state(args, state)
        print(f"Compile submitted: {job.url}", flush=True)
    else:
        job = hub.get_job(state["compile_job_id"])
        print(f"Compile resumed: {job.url}", flush=True)
    compiled_model_id = _require_target(job, "Compile")
    if state.get("compiled_model_id", compiled_model_id) != compiled_model_id:
        raise ValueError("Recorded compiled model ID changed")
    state["compiled_model_id"] = compiled_model_id
    _save_state(args, state)
    return compiled_model_id, job


def _probe(
    args: argparse.Namespace,
    spec: dict[str, Any],
    calibration_graph: dict[str, Any],
    model_id: str,
    compile_job: Any,
    device: dict[str, str],
) -> dict[str, Any]:
    samples = [
        next(
            sample for sample in calibration_graph["samples"]
            if sample["source_id"] == args.source_id and sample["frame"] == frame
        )
        for frame in (0, 1)
    ]
    feeds = [
        _sample_feed(args.calibration_dir, sample, spec["input_names"])
        for sample in samples
    ]
    session = _session(args.onnx_dir / spec["onnx"])
    expected = [session.run(spec["output_names"], feed) for feed in feeds]
    input_order = list(compile_job.get_target_shapes())
    if set(input_order) != set(spec["input_names"]):
        raise ValueError(f"Compiled input names changed: {input_order}")
    actual = _cloud_batch(
        model_id=model_id,
        device=device,
        samples=feeds,
        input_order=input_order,
        output_names=spec["output_names"],
        stem=args.output_dir / "two_frame_probe",
        retry_failed=args.retry_failed,
    )
    report: dict[str, Any] = {"format": FORMAT, "graph": args.graph, "samples": []}
    passed = True
    for sample, feed, received_outputs, expected_outputs in zip(
        samples, feeds, actual, expected, strict=True
    ):
        outputs = {}
        position = int(feed["position"].item())
        if position != sample["frame"]:
            raise ValueError("Captured sample position differs from frame")
        for name, received, reference in zip(
            spec["output_names"], received_outputs, expected_outputs, strict=True
        ):
            if name == "output_hidden":
                metrics = _probe_metrics(received, reference)
                passed &= bool(
                    metrics["finite"] and metrics["relative_rms"] <= MAX_RELATIVE_RMS
                )
            else:
                if received.shape != reference.shape or received.ndim < 3:
                    raise ValueError(f"Unexpected cache output shape: {name}")
                slot = position % received.shape[2]
                written = _probe_metrics(received[:, :, slot], reference[:, :, slot])
                preserved_max_abs: float | None = 0.0
                previous = feed[name.removeprefix("output_")]
                if not np.isfinite(received).all():
                    preserved_max_abs = None
                for lower, upper in ((0, slot), (slot + 1, received.shape[2])):
                    if lower < upper and preserved_max_abs is not None:
                        delta = received[:, :, lower:upper] - previous[:, :, lower:upper]
                        preserved_max_abs = max(
                            preserved_max_abs, float(np.abs(delta).max())
                        )
                metrics = {**written,
                    "written_slot": slot,
                    "written_slot_actual_nonzero": int(np.count_nonzero(received[:, :, slot])),
                    "written_slot_reference_nonzero": int(np.count_nonzero(reference[:, :, slot])),
                    "written_slot_metrics": written,
                    "preserved_max_abs": preserved_max_abs,
                }
                passed &= bool(
                    written["finite"]
                    and written["relative_rms"] <= MAX_RELATIVE_RMS
                    and metrics["written_slot_actual_nonzero"] > 0
                    and preserved_max_abs is not None
                    and preserved_max_abs <= MAX_PRESERVED_ABS
                )
            outputs[name] = metrics
        report["samples"].append({
            "source_id": sample["source_id"], "frame": sample["frame"],
            "outputs": outputs,
        })
    report["single_graph_gate_pass"] = passed
    report["gate"] = {
        "max_relative_rms": MAX_RELATIVE_RMS,
        "max_preserved_abs": MAX_PRESERVED_ABS,
    }
    _write_json(args.output_dir / "quantized_report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-dir", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--patch-dir", type=Path, required=True)
    parser.add_argument("--source-quantization-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--graph", default="temporal_layers_30_31")
    parser.add_argument("--source-id", default="clean-000")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument(
        "--retry-failed", action="store_true",
        help="Replace failed recorded jobs without re-uploading successful models",
    )
    args = parser.parse_args()
    if args.frame != 0:
        parser.error("This experiment probes captured frames 0 and 1")

    config, spec, baseline, patch_path = _patch_and_baseline(args)
    patch_sha = _sha256(patch_path)
    options = _quantize_options(Precision.w8a16_mixed_fp16, 100.0)
    plan = {
        "format": FORMAT,
        "graph": args.graph,
        "patch_sha256": patch_sha,
        "baseline_manifest_sha256": _sha256(args.source_quantization_manifest),
        "calibration_manifest_sha256": config["calibration_manifest_sha256"],
        "target_device": config["target_device"],
        "quantize_options": options,
        "compile_options": _compile_options(),
    }
    state = _job_state(args, plan)
    _source_model_id(args, state, patch_path, patch_sha)
    calibration = json.loads((args.calibration_dir / "manifest.json").read_text())
    compiled_id, compile_job = _quantize_and_compile(
        args, state, calibration["graphs"][args.graph], config["target_device"], options
    )
    report = _probe(
        args, spec, calibration["graphs"][args.graph], compiled_id,
        compile_job, config["target_device"]
    )
    if not report["single_graph_gate_pass"]:
        raise RuntimeError(
            "Quantized shard did not pass the two-frame direct probe; "
            "no candidate manifest was created"
        )

    candidate = copy.deepcopy(baseline)
    candidate.pop("preflight_receipt", None)
    record = candidate["graphs"][args.graph]
    for stale_key in (
        "source_reused_by_sha256", "quantize_live_status", "compile_live_status", "error"
    ):
        record.pop(stale_key, None)
    record.update({
        "onnx_sha256": patch_sha,
        "reference_onnx_sha256": config["onnx_sha256"],
        "source_model_id": state["source_model_id"],
        "quantize_job_id": state["quantize_job_id"],
        "quantized_model_id": state["quantized_model_id"],
        "compile_job_id": state["compile_job_id"],
        "compiled_model_id": compiled_id,
        "precision": str(Precision.w8a16_mixed_fp16),
        "litemp_percentage": 100.0,
        "quantize_options": options,
        "compile_runtime": COMPILE_RUNTIME.value,
        "compile_options": _compile_options(),
        "status": "compile_succeeded",
        "dynamic_rmsnorm_patch": str(patch_path),
        "single_graph_probe_report": str(args.output_dir / "quantized_report.json"),
    })
    candidate.setdefault("graph_litemp_percentages", {})[args.graph] = 100.0
    candidate.setdefault("graph_precisions", {})[args.graph] = str(
        Precision.w8a16_mixed_fp16
    )
    candidate["single_graph_override"] = {
        "graph": args.graph,
        "baseline_manifest_sha256": plan["baseline_manifest_sha256"],
        "patched_onnx_sha256": patch_sha,
        "validation": "two-frame-direct-probe-only; chained parity pending",
    }
    candidate_path = args.output_dir / "quantization_manifest.json"
    _write_json(candidate_path, candidate)
    print(
        f"Single-graph gate PASS; candidate manifest={candidate_path}. "
        "Run chained validation before deployment claims.",
        flush=True,
    )


if __name__ == "__main__":
    main()

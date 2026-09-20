"""Test the codebook-0 one-valid-slot attention identity on the failing chain.

``prepare`` exports only DepFormer codebook 0 from the real checkpoint and
requires FP32 ONNX parity on captured and failing-chain inputs. ``cloud``
reuses that receipt, quantizes with the *existing* codebook-0 settings, compiles
one DLC, and probes the two saved head outputs. Other graphs are untouched.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import qai_hub as hub
import torch

from moshi_export_lm_onnx import _export
from moshi_probe_dynamic_norm_cloud import _sample_feed
from moshi_probe_float_compile_cloud import _source_and_sample
from moshi_probe_quantized_graph_cloud import _probe_metrics, _session
from moshi_quantize_lm_graph_set import (
    Precision,
    _compile_options,
    _input_specs,
    _load_calibration_entries,
    _quantize_options,
)
from moshi_verify_lm_graph_set_cloud import _cloud_batch, _sha256, _write_json

from qai_hub_models.models.templates.moshi.explicit_cache import (
    ExplicitDepFormer,
    ExplicitDepFormerStep,
)
from qai_hub_models.models.templates.moshi.model import load_moshi_lm


FORMAT = "moshi-depformer0-single-value-v1"
GRAPH = "depformer_codebook_0"
MAX_RELATIVE_RMS = 0.05
MAX_PRESERVED_ABS = 0.001


def _paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    source = args.onnx_dir / f"{GRAPH}.onnx"
    patched = args.output_dir / f"{GRAPH}.onnx"
    head = args.validation_dir / "017_head.npz"
    return source, patched, head


def _chain_feeds(args: argparse.Namespace, spec: dict[str, Any]) -> list[dict[str, np.ndarray]]:
    depformer_record = json.loads(
        (args.validation_dir / "018_depformer_codebook_0.json").read_text()
    )
    baseline = json.loads(args.source_quantization_manifest.read_text())
    if depformer_record.get("job_id") != args.expected_depformer_job_id:
        raise ValueError(
            "Validation directory has a different DepFormer-0 job: "
            f"{depformer_record.get('job_id')} != {args.expected_depformer_job_id}"
        )
    if depformer_record.get("target_model_id") != baseline["graphs"][GRAPH].get(
        "compiled_model_id"
    ):
        raise ValueError("Validation job used a different DepFormer-0 model")
    if depformer_record.get("samples") != 2:
        raise ValueError("Validation DepFormer-0 job must contain two samples")
    run_path = args.validation_dir / "run_manifest.json"
    run = json.loads(run_path.read_text())
    if run.get("frames") != 2 or [
        item["id"] for item in run.get("sources", [])
    ] != [args.source_id]:
        raise ValueError(
            "Expected exactly two frames for the selected source "
            "in the validation directory"
        )
    _, _, head_path = _paths(args)
    feeds = []
    with np.load(head_path) as arrays:
        for frame in (0, 1):
            temporal = np.array(arrays[f"sample_{frame}_output_0"], dtype=np.float32)
            logits = arrays[f"sample_{frame}_output_1"]
            if not np.isfinite(temporal).all() or not np.isfinite(logits).all():
                raise ValueError(f"Head output contains NaN/Inf at frame {frame}")
            token = np.argmax(logits, axis=-1)[:, 0, 0].astype(np.int64)
            values = [
                token,
                temporal,
                *(np.zeros(shape, np.float32) for shape in spec["cache_shapes"]),
            ]
            feeds.append(dict(zip(spec["input_names"], values, strict=True)))
    return feeds


def _calibration_feeds(
    args: argparse.Namespace, spec: dict[str, Any]
) -> list[dict[str, np.ndarray]]:
    calibration = json.loads((args.calibration_dir / "manifest.json").read_text())
    samples = calibration["graphs"][GRAPH]["samples"]
    feeds = []
    for frame in (0, 1):
        sample = next(
            (
                item
                for item in samples
                if item["source_id"] == args.source_id and item["frame"] == frame
            ),
            None,
        )
        if sample is None:
            raise ValueError(f"Missing calibration source={args.source_id} frame={frame}")
        feeds.append(_sample_feed(args.calibration_dir, sample, spec["input_names"]))
    return feeds


def _cpu_parity(
    original_path: Path,
    patched_path: Path,
    spec: dict[str, Any],
    feeds: list[dict[str, np.ndarray]],
    label: str,
) -> list[dict[str, Any]]:
    original = _session(original_path)
    patched = _session(patched_path)
    reports = []
    for frame, feed in enumerate(feeds):
        expected = original.run(spec["output_names"], feed)
        received = patched.run(spec["output_names"], feed)
        outputs = {}
        for name, actual, reference in zip(
            spec["output_names"], received, expected, strict=True
        ):
            if actual.shape != reference.shape or not np.isfinite(actual).all():
                raise ValueError(f"CPU {label} frame={frame} invalid output {name}")
            if name == "audio_token":
                np.testing.assert_array_equal(actual, reference)
            else:
                np.testing.assert_allclose(actual, reference, rtol=2e-4, atol=1e-4)
            outputs[name] = _probe_metrics(actual, reference)
        reports.append({"set": label, "frame": frame, "outputs": outputs})
        print(
            f"CPU {label} frame={frame}: logits rel RMS="
            f"{outputs['audio_logits']['relative_rms']:.6g}",
            flush=True,
        )
    return reports


def _prepare(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise ValueError("Use a new empty output directory for prepare")
    config, spec, _ = _source_and_sample(args)
    original_path, patched_path, head_path = _paths(args)
    if not head_path.is_file():
        raise FileNotFoundError(head_path)
    chain_feeds = _chain_feeds(args, spec)
    calibration_feeds = _calibration_feeds(args, spec)

    print(f"Loading real checkpoint: {args.model_dir}", flush=True)
    lm = load_moshi_lm(model_dir=args.model_dir, device=args.device)
    depformer = ExplicitDepFormer(lm)
    # This is the only step exported here, so moving the referenced modules to
    # CPU is safe and avoids briefly duplicating a large BF16 GPU model.
    module = ExplicitDepFormerStep(
        depformer, 0, single_value_attention=True
    ).cpu().float().eval()
    del depformer, lm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    inputs = tuple(torch.from_numpy(calibration_feeds[0][name]) for name in spec["input_names"])
    _export(module, inputs, patched_path, spec["input_names"], spec["output_names"])
    del module
    gc.collect()

    reports = _cpu_parity(original_path, patched_path, spec, calibration_feeds, "calibration")
    reports.extend(_cpu_parity(original_path, patched_path, spec, chain_feeds, "failing_chain"))
    receipt = {
        "format": FORMAT,
        "graph": GRAPH,
        "source_onnx_sha256": _sha256(original_path),
        "patched_onnx_sha256": _sha256(patched_path),
        "graph_manifest_sha256": config["graph_manifest_sha256"],
        "calibration_manifest_sha256": config["calibration_manifest_sha256"],
        "baseline_manifest_sha256": _sha256(args.source_quantization_manifest),
        "head_archive_sha256": _sha256(head_path),
        "failing_depformer_job_id": args.expected_depformer_job_id,
        "cpu_samples": reports,
    }
    _write_json(args.output_dir / "cpu_pass.json", receipt)
    print(f"CPU parity PASS; codebook-0-only ONNX: {patched_path}", flush=True)


def _verified_plan(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path]:
    config, spec, _ = _source_and_sample(args)
    _chain_feeds(args, spec)
    original_path, patched_path, head_path = _paths(args)
    receipt = json.loads((args.output_dir / "cpu_pass.json").read_text())
    checks = {
        "format": FORMAT,
        "graph": GRAPH,
        "source_onnx_sha256": _sha256(original_path),
        "patched_onnx_sha256": _sha256(patched_path),
        "graph_manifest_sha256": config["graph_manifest_sha256"],
        "calibration_manifest_sha256": config["calibration_manifest_sha256"],
        "baseline_manifest_sha256": _sha256(args.source_quantization_manifest),
        "head_archive_sha256": _sha256(head_path),
        "failing_depformer_job_id": args.expected_depformer_job_id,
    }
    if any(receipt.get(key) != value for key, value in checks.items()):
        raise ValueError("CPU parity receipt no longer matches the ONNX/manifests/head outputs")
    if len(receipt.get("cpu_samples", [])) != 4:
        raise ValueError("CPU parity receipt does not cover four samples")
    baseline = json.loads(args.source_quantization_manifest.read_text())
    if any(not item.get("compiled_model_id") for item in baseline["graphs"].values()):
        raise ValueError("Baseline needs a compiled model ID for every graph")
    record = baseline["graphs"][GRAPH]
    precision = Precision.parse(record["precision"])
    if not precision.activations_type or not precision.weights_type:
        raise ValueError("Codebook 0 baseline is not quantized")
    options = _quantize_options(precision, float(record["litemp_percentage"]))
    if options != record["quantize_options"]:
        raise ValueError("Baseline codebook-0 quantize options do not match precision")
    plan = {
        **checks,
        "target_device": config["target_device"],
        "precision": record["precision"],
        "quantize_options": options,
        "compile_options": _compile_options(),
    }
    return plan, spec, baseline, patched_path


def _save_state(args: argparse.Namespace, state: dict[str, Any]) -> None:
    _write_json(args.output_dir / "jobs.json", state)


def _target(job: Any, stage: str) -> str:
    job.wait()
    status = job.get_status()
    if not status.success:
        raise RuntimeError(f"{stage} failed ({status}): {job.url}")
    model = job.get_target_model()
    if model is None:
        raise RuntimeError(f"{stage} returned no model: {job.url}")
    return model.model_id


def _retry_failed(
    args: argparse.Namespace, state: dict[str, Any], job_key: str, model_key: str
) -> None:
    job_id = state.get(job_key)
    if not job_id:
        return
    job = hub.get_job(job_id)
    if not job.get_status().failure:
        return
    if not args.retry_failed:
        raise RuntimeError(f"Recorded {job_key} failed: {job.url}; pass --retry-failed")
    state.setdefault("failed_job_ids", []).append(job_id)
    state.pop(job_key, None)
    state.pop(model_key, None)
    _save_state(args, state)


def _cloud(args: argparse.Namespace) -> None:
    plan, spec, baseline, patched_path = _verified_plan(args)
    state_path = args.output_dir / "jobs.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"plan": plan}
    if state.get("plan") != plan:
        raise ValueError("Cloud plan changed; use a new output directory")
    _save_state(args, state)
    if not state.get("source_model_id"):
        state["source_model_id"] = hub.upload_model(str(patched_path)).model_id
        _save_state(args, state)
        print(f"Patched source uploaded: {state['source_model_id']}", flush=True)

    calibration = json.loads((args.calibration_dir / "manifest.json").read_text())
    calibration_graph = calibration["graphs"][GRAPH]
    precision = Precision.parse(plan["precision"])
    _retry_failed(args, state, "quantize_job_id", "quantized_model_id")
    if not state.get("quantize_job_id"):
        entries = _load_calibration_entries(args.calibration_dir, calibration_graph)
        job = hub.submit_quantize_job(
            model=hub.get_model(state["source_model_id"]),
            calibration_data=entries,
            activations_dtype=precision.activations_type,
            weights_dtype=precision.weights_type,
            name="moshi-depformer-codebook-0-single-value",
            options=plan["quantize_options"],
        )
        state["quantize_job_id"] = job.job_id
        _save_state(args, state)
        print(f"Quantize submitted: {job.url}", flush=True)
        del entries
    else:
        job = hub.get_job(state["quantize_job_id"])
        print(f"Quantize resumed: {job.url}", flush=True)
    quantized_id = _target(job, "Quantize")
    if state.get("quantized_model_id", quantized_id) != quantized_id:
        raise ValueError("Recorded quantized model ID changed")
    state["quantized_model_id"] = quantized_id
    _save_state(args, state)

    _retry_failed(args, state, "compile_job_id", "compiled_model_id")
    if not state.get("compile_job_id"):
        job = hub.submit_compile_job(
            model=hub.get_model(quantized_id),
            input_specs=_input_specs(args.calibration_dir, calibration_graph),
            device=hub.Device(**plan["target_device"]),
            name="moshi-depformer-codebook-0-single-value-dlc",
            options=plan["compile_options"],
        )
        state["compile_job_id"] = job.job_id
        _save_state(args, state)
        print(f"Compile submitted: {job.url}", flush=True)
    else:
        job = hub.get_job(state["compile_job_id"])
        print(f"Compile resumed: {job.url}", flush=True)
    compiled_id = _target(job, "Compile")
    if state.get("compiled_model_id", compiled_id) != compiled_id:
        raise ValueError("Recorded compiled model ID changed")
    state["compiled_model_id"] = compiled_id
    _save_state(args, state)

    feeds = _chain_feeds(args, spec)
    session = _session(patched_path)
    expected = [session.run(spec["output_names"], feed) for feed in feeds]
    input_order = list(job.get_target_shapes())
    if set(input_order) != set(spec["input_names"]):
        raise ValueError(f"Compiled input names changed: {input_order}")
    actual = _cloud_batch(
        model_id=compiled_id,
        device=plan["target_device"],
        samples=feeds,
        input_order=input_order,
        output_names=spec["output_names"],
        stem=args.output_dir / "two_frame_probe",
        retry_failed=args.retry_failed,
    )
    report: dict[str, Any] = {
        "format": FORMAT,
        "graph": GRAPH,
        "source_model_id": state["source_model_id"],
        "quantized_model_id": quantized_id,
        "compiled_model_id": compiled_id,
        "quantize_job_id": state["quantize_job_id"],
        "compile_job_id": state["compile_job_id"],
        "samples": [],
    }
    passed = True
    for frame, (feed, received, reference) in enumerate(zip(feeds, actual, expected, strict=True)):
        logits = _probe_metrics(received[1], reference[1])
        token_match = bool(np.array_equal(received[0], reference[0]))
        cache = {}
        cache_pass = True
        for name, cloud_value, cpu_value in zip(
            spec["output_names"][2:], received[2:], reference[2:], strict=True
        ):
            if cloud_value.shape != cpu_value.shape or cloud_value.ndim != 4:
                raise ValueError(f"Unexpected cache shape for {name}")
            written = _probe_metrics(cloud_value[:, :, 0], cpu_value[:, :, 0])
            preserved = cloud_value[:, :, 1:] - feed[name.removeprefix("output_")][:, :, 1:]
            preserved_max = float(np.max(np.abs(preserved))) if preserved.size else 0.0
            cache[name] = {"written": written, "preserved_max_abs": preserved_max}
            cache_pass &= bool(
                written["finite"]
                and written["relative_rms"] <= MAX_RELATIVE_RMS
                and np.isfinite(preserved_max)
                and preserved_max <= MAX_PRESERVED_ABS
            )
        sample_pass = bool(
            token_match
            and logits["finite"]
            and logits["relative_rms"] <= MAX_RELATIVE_RMS
            and cache_pass
        )
        passed &= sample_pass
        report["samples"].append({
            "frame": frame,
            "previous_token": int(feed["previous_token"].item()),
            "audio_token_match": token_match,
            "audio_logits": logits,
            "cache": cache,
            "passed": sample_pass,
        })
        print(
            f"frame={frame} token_match={token_match} logits={logits} "
            f"cache_pass={cache_pass}",
            flush=True,
        )
    report["single_graph_gate_pass"] = passed
    report["gate"] = {
        "max_relative_rms": MAX_RELATIVE_RMS,
        "max_preserved_abs": MAX_PRESERVED_ABS,
    }
    _write_json(args.output_dir / "cloud_report.json", report)
    if not passed:
        print("Single-graph gate FAIL; no candidate manifest created", flush=True)
        return

    candidate = copy.deepcopy(baseline)
    record = candidate["graphs"][GRAPH]
    record.update({
        "onnx_sha256": plan["patched_onnx_sha256"],
        "reference_onnx_sha256": plan["source_onnx_sha256"],
        "source_model_id": state["source_model_id"],
        "quantize_job_id": state["quantize_job_id"],
        "quantized_model_id": quantized_id,
        "compile_job_id": state["compile_job_id"],
        "compiled_model_id": compiled_id,
        "status": "compile_succeeded",
        "single_value_attention_patch": str(patched_path),
        "single_graph_probe_report": str(args.output_dir / "cloud_report.json"),
    })
    candidate["depformer0_single_value_override"] = {
        "baseline_manifest_sha256": plan["baseline_manifest_sha256"],
        "patched_onnx_sha256": plan["patched_onnx_sha256"],
        "validation": "two-frame direct probe only; chained parity pending",
    }
    candidate_path = args.output_dir / "quantization_manifest.json"
    _write_json(candidate_path, candidate)
    print(f"Single-graph gate PASS; candidate manifest={candidate_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "cloud"))
    parser.add_argument("--onnx-dir", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--source-quantization-manifest", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--expected-depformer-job-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, help="Real checkpoint, needed only by prepare")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--source-id", default="clean-000")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    args.graph = GRAPH
    args.frame = 0
    if args.mode == "prepare":
        if args.model_dir is None:
            parser.error("prepare requires --model-dir")
        _prepare(args)
    else:
        _cloud(args)


if __name__ == "__main__":
    main()

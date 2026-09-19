"""Test a rescaled RMSNorm in one real Temporal shard before re-quantizing it.

The prepare stage only uses local ONNX Runtime. The cloud stage uploads and
compiles the patched shard once, then probes the captured frame-0 input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import qai_hub as hub

from moshi_probe_float_compile_cloud import _compiled_model, _source_and_sample
from moshi_probe_quantized_graph_cloud import _probe_metrics, _session
from moshi_verify_dynamic_shard import patch_dynamic_rmsnorms
from moshi_verify_lm_graph_set_cloud import _cloud_batch, _sha256, _write_json


FORMAT = "moshi-lm-dynamic-rmsnorm-probe-v1"


def _sample_feed(
    calibration_dir: Path, sample: dict[str, Any], names: list[str]
) -> dict[str, np.ndarray]:
    with np.load(calibration_dir / sample["file"]) as arrays:
        return {name: np.array(arrays[name], copy=True) for name in names}


def _second_sample(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads((args.calibration_dir / "manifest.json").read_text())
    return next(
        sample
        for sample in manifest["graphs"][args.graph]["samples"]
        if sample["source_id"] == args.source_id and sample["frame"] == 1
    )


def _cpu(args: argparse.Namespace) -> None:
    if args.frame != 0:
        raise ValueError("The dynamic-norm control starts from captured frame 0")
    config, spec, first = _source_and_sample(args)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Use an empty output directory; preserve earlier attempts")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_path = args.onnx_dir / spec["onnx"]
    patched_path = args.output_dir / f"{args.graph}.onnx"
    model = onnx.load(str(source_path))
    changed = patch_dynamic_rmsnorms(model)
    onnx.save(model, str(patched_path))
    del model

    original = _session(source_path)
    patched = _session(patched_path)
    reports = []
    for sample in (first, _second_sample(args)):
        feed = _sample_feed(args.calibration_dir, sample, spec["input_names"])
        if int(feed["position"].item()) != sample["frame"]:
            raise ValueError("Calibration sample position differs from frame")
        hidden = feed["hidden"]
        with np.errstate(over="ignore"):
            squared = hidden.astype(np.float16) ** 2
        expected = original.run(spec["output_names"], feed)
        actual = patched.run(spec["output_names"], feed)
        outputs = {}
        for name, received, reference in zip(
            spec["output_names"], actual, expected, strict=True
        ):
            if name == "output_hidden":
                np.testing.assert_allclose(received, reference, rtol=1e-4, atol=1e-5)
                outputs[name] = _probe_metrics(received, reference)
                continue
            slot = sample["frame"] % received.shape[2]
            if not np.array_equal(received[:, :, :slot], reference[:, :, :slot]):
                raise ValueError(f"CPU parity changed preserved cache before slot: {name}")
            if not np.array_equal(received[:, :, slot + 1 :], reference[:, :, slot + 1 :]):
                raise ValueError(f"CPU parity changed preserved cache after slot: {name}")
            written = received[:, :, slot]
            reference_written = reference[:, :, slot]
            np.testing.assert_allclose(
                written, reference_written, rtol=1e-4, atol=1e-5
            )
            outputs[name] = _probe_metrics(written, reference_written)
        reports.append({
            "frame": sample["frame"],
            "input_hidden_absmax": float(np.abs(hidden).max()),
            "input_fp16_square_inf": int(np.count_nonzero(np.isinf(squared))),
            "outputs": outputs,
        })
        print(
            f"CPU frame={sample['frame']}: FP16-square Inf="
            f"{reports[-1]['input_fp16_square_inf']} "
            f"hidden rel RMS={outputs['output_hidden']['relative_rms']:.6g}",
            flush=True,
        )

    receipt = {
        "format": FORMAT,
        "graph": args.graph,
        "source_onnx_sha256": _sha256(source_path),
        "patched_onnx_sha256": _sha256(patched_path),
        "graph_manifest_sha256": config["graph_manifest_sha256"],
        "calibration_manifest_sha256": config["calibration_manifest_sha256"],
        "norms": changed,
        "cpu_samples": reports,
    }
    _write_json(args.output_dir / "cpu_pass.json", receipt)
    print(f"CPU parity PASS; patched shard: {patched_path}", flush=True)


def _cloud(args: argparse.Namespace) -> None:
    config, spec, sample = _source_and_sample(args)
    receipt_path = args.output_dir / "cpu_pass.json"
    if not receipt_path.exists():
        raise ValueError("Run prepare first and pass CPU parity before uploading")
    receipt = json.loads(receipt_path.read_text())
    source_path = args.onnx_dir / spec["onnx"]
    patched_path = args.output_dir / f"{args.graph}.onnx"
    if (
        receipt.get("format") != FORMAT
        or receipt.get("graph") != args.graph
        or receipt.get("source_onnx_sha256") != _sha256(source_path)
        or receipt.get("patched_onnx_sha256") != _sha256(patched_path)
        or receipt.get("graph_manifest_sha256") != config["graph_manifest_sha256"]
        or receipt.get("calibration_manifest_sha256")
        != config["calibration_manifest_sha256"]
    ):
        raise ValueError("Model or manifests changed after CPU parity; use a new run")
    feed = _sample_feed(args.calibration_dir, sample, spec["input_names"])
    upload_path = args.output_dir / "upload.json"
    upload = json.loads(upload_path.read_text()) if upload_path.exists() else {}
    if upload and upload.get("patched_onnx_sha256") != receipt["patched_onnx_sha256"]:
        raise ValueError("Uploaded model digest does not match patched ONNX")
    if not upload.get("source_model_id"):
        model_id = hub.upload_model(str(patched_path)).model_id
        upload = {
            "patched_onnx_sha256": receipt["patched_onnx_sha256"],
            "source_model_id": model_id,
        }
        _write_json(upload_path, upload)
        print(f"Patched shard uploaded: {model_id}", flush=True)

    compile_config = {
        **config,
        "format": FORMAT,
        "onnx_sha256": receipt["patched_onnx_sha256"],
        "source_model_id": upload["source_model_id"],
    }
    input_specs = {
        name: (tuple(value.shape), str(value.dtype)) for name, value in feed.items()
    }
    compiled_model_id, job = _compiled_model(
        compile_config, args.output_dir / "compile.json", input_specs
    )
    input_order = list(job.get_target_shapes())
    if set(input_order) != set(spec["input_names"]):
        raise ValueError(f"Compiled input names changed: {input_order}")
    expected = _session(source_path).run(spec["output_names"], feed)
    actual = _cloud_batch(
        model_id=compiled_model_id,
        device=config["target_device"],
        samples=[feed],
        input_order=input_order,
        output_names=spec["output_names"],
        stem=args.output_dir / "frame_0_probe",
        retry_failed=False,
    )[0]
    outputs = {}
    position = int(feed["position"].item())
    for name, received, reference in zip(
        spec["output_names"], actual, expected, strict=True
    ):
        metrics = _probe_metrics(received, reference)
        metrics["actual_nonzero"] = int(np.count_nonzero(received))
        metrics["reference_nonzero"] = int(np.count_nonzero(reference))
        if name != "output_hidden" and received.ndim >= 3:
            slot = position % received.shape[2]
            metrics["written_slot"] = slot
            metrics["written_slot_actual_nonzero"] = int(
                np.count_nonzero(received[:, :, slot])
            )
            metrics["written_slot_reference_nonzero"] = int(
                np.count_nonzero(reference[:, :, slot])
            )
            metrics["written_slot_metrics"] = _probe_metrics(
                received[:, :, slot], reference[:, :, slot]
            )
        outputs[name] = metrics
    report = {
        "format": FORMAT,
        "graph": args.graph,
        "source_model_id": upload["source_model_id"],
        "compiled_model_id": compiled_model_id,
        "compile_job_id": job.job_id,
        "frame": sample["frame"],
        "outputs": outputs,
    }
    _write_json(args.output_dir / "cloud_report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    print("Inspect hidden and written-slot errors before quantizing", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "cloud"))
    parser.add_argument("--onnx-dir", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--source-quantization-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--graph", default="temporal_layers_30_31")
    parser.add_argument("--source-id", default="clean-000")
    parser.add_argument("--frame", type=int, default=0)
    args = parser.parse_args()
    if args.frame != 0:
        parser.error("This control uses captured frame 0, followed by frame 1 on CPU")
    if not args.graph.startswith("temporal_layers_"):
        parser.error("Choose a Temporal shard")
    if args.mode == "prepare":
        _cpu(args)
    else:
        _cloud(args)


if __name__ == "__main__":
    main()

"""Probe one compiled Moshi graph directly with captured calibration inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from moshi_verify_lm_graph_set_cloud import (
    CALIBRATION_FORMAT,
    GRAPH_SET_FORMAT,
    QUANTIZATION_FORMAT,
    _cloud_batch,
    _compiled_input_order,
    _metrics,
    _target_id,
    _write_json,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _graph_specs(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    specs = {"frontend": manifest["frontend"], "head": manifest["head"]}
    specs.update(
        (Path(item["onnx"]).stem, item) for item in manifest["temporal_shards"]
    )
    specs.update(
        (Path(item["onnx"]).stem, item) for item in manifest["depformer"]["steps"]
    )
    return specs


def _session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def _finite_counts(value: np.ndarray) -> dict[str, int]:
    array = np.asarray(value)
    return {
        "elements": int(array.size),
        "finite": int(np.count_nonzero(np.isfinite(array))),
        "nan": int(np.count_nonzero(np.isnan(array))),
        "positive_inf": int(np.count_nonzero(np.isposinf(array))),
        "negative_inf": int(np.count_nonzero(np.isneginf(array))),
    }


def _probe_metrics(actual: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    actual_counts = _finite_counts(actual)
    reference_counts = _finite_counts(reference)
    finite = (
        actual_counts["finite"] == actual_counts["elements"]
        and reference_counts["finite"] == reference_counts["elements"]
    )
    if finite:
        return {"finite": True, **_metrics(actual, reference)}
    return {
        "finite": False,
        "relative_rms": None,
        "rmse": None,
        "max_abs": None,
        "actual_counts": actual_counts,
        "reference_counts": reference_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-dir", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--quantization-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--graph", default="temporal_layers_2_3")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    if args.samples < 1:
        raise SystemExit("--samples must be positive")

    graph_path = args.onnx_dir / "manifest.json"
    calibration_path = args.calibration_dir / "manifest.json"
    quantization_path = args.quantization_dir / "quantization_manifest.json"
    graph_manifest = json.loads(graph_path.read_text())
    calibration = json.loads(calibration_path.read_text())
    quantization = json.loads(quantization_path.read_text())
    if graph_manifest.get("format") != GRAPH_SET_FORMAT:
        raise SystemExit(f"Unsupported graph manifest: {graph_manifest.get('format')}")
    if calibration.get("format") != CALIBRATION_FORMAT:
        raise SystemExit(
            f"Unsupported calibration manifest: {calibration.get('format')}"
        )
    if quantization.get("format") != QUANTIZATION_FORMAT:
        raise SystemExit(
            f"Unsupported quantization manifest: {quantization.get('format')}"
        )
    if calibration.get("graph_manifest_sha256") != _sha256(graph_path):
        raise SystemExit("Calibration does not match the ONNX graph manifest")
    if quantization.get("graph_manifest_sha256") != _sha256(graph_path):
        raise SystemExit("Quantized models do not match the ONNX graph manifest")
    if quantization.get("calibration_manifest_sha256") != _sha256(calibration_path):
        raise SystemExit("Quantized models do not match the calibration manifest")

    spec = _graph_specs(graph_manifest).get(args.graph)
    if spec is None:
        raise SystemExit(f"Unknown graph: {args.graph}")
    calibration_graph = calibration["graphs"].get(args.graph)
    if calibration_graph is None:
        raise SystemExit(f"Calibration is missing graph {args.graph}")
    samples = calibration_graph.get("samples", [])[: args.samples]
    if len(samples) < args.samples:
        raise SystemExit(
            f"Graph {args.graph} has {len(samples)} calibration samples; "
            f"requested {args.samples}"
        )

    feeds: list[dict[str, np.ndarray]] = []
    for sample in samples:
        with np.load(args.calibration_dir / sample["file"]) as arrays:
            feeds.append(
                {
                    name: np.array(arrays[name], copy=True)
                    for name in spec["input_names"]
                }
            )

    session = _session(args.onnx_dir / spec["onnx"])
    expected = [session.run(spec["output_names"], feed) for feed in feeds]
    device = quantization.get("target_device")
    if not isinstance(device, dict) or not device.get("name"):
        raise SystemExit("Quantization manifest has no target device")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    actual = _cloud_batch(
        model_id=_target_id(quantization, args.graph),
        device=device,
        samples=feeds,
        input_order=_compiled_input_order(quantization, args.graph),
        output_names=spec["output_names"],
        stem=args.output_dir / args.graph,
        retry_failed=args.retry_failed,
    )
    report: dict[str, Any] = {
        "format": "moshi-lm-quantized-graph-probe-v1",
        "graph": args.graph,
        "samples": [
            {
                "source_id": sample["source_id"],
                "frame": sample["frame"],
                "outputs": {
                    name: _probe_metrics(received, reference)
                    for name, received, reference in zip(
                        spec["output_names"], cloud_outputs, cpu_outputs, strict=True
                    )
                },
            }
            for sample, cloud_outputs, cpu_outputs in zip(
                samples, actual, expected, strict=True
            )
        ],
    }
    nonfinite_outputs = [
        {
            "source_id": sample["source_id"],
            "frame": sample["frame"],
            "output": name,
            "actual_counts": metrics["actual_counts"],
            "reference_counts": metrics["reference_counts"],
        }
        for sample in report["samples"]
        for name, metrics in sample["outputs"].items()
        if not metrics["finite"]
    ]
    report["nonfinite_outputs"] = nonfinite_outputs
    report["passed"] = not nonfinite_outputs
    _write_json(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    if nonfinite_outputs:
        raise RuntimeError(
            f"Cloud output contains NaN or Inf; inspect "
            f"{args.output_dir / 'report.json'}"
        )
    print(f"Single-graph cloud probe: PASS graph={args.graph}", flush=True)


if __name__ == "__main__":
    main()

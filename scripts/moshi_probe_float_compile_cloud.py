"""Compile one uploaded, unquantized Moshi ONNX graph and probe one frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import qai_hub as hub
from moshi_probe_quantized_graph_cloud import _graph_specs, _probe_metrics, _session
from moshi_quantize_lm_graph_set import _compile_options
from moshi_verify_lm_graph_set_cloud import (
    CALIBRATION_FORMAT,
    GRAPH_SET_FORMAT,
    QUANTIZATION_FORMAT,
    _cloud_batch,
    _sha256,
    _write_json,
)

FORMAT = "moshi-lm-unquantized-compile-probe-v1"


def _source_and_sample(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    graph_path = args.onnx_dir / "manifest.json"
    calibration_path = args.calibration_dir / "manifest.json"
    source_path = args.source_quantization_manifest
    graph_manifest = json.loads(graph_path.read_text())
    calibration = json.loads(calibration_path.read_text())
    source = json.loads(source_path.read_text())
    if graph_manifest.get("format") != GRAPH_SET_FORMAT:
        raise ValueError(f"Unsupported ONNX graph manifest: {graph_path}")
    if calibration.get("format") != CALIBRATION_FORMAT:
        raise ValueError(f"Unsupported calibration manifest: {calibration_path}")
    if source.get("format") != QUANTIZATION_FORMAT:
        raise ValueError(f"Unsupported source manifest: {source_path}")
    graph_hash = _sha256(graph_path)
    calibration_hash = _sha256(calibration_path)
    if calibration.get("graph_manifest_sha256") != graph_hash:
        raise ValueError("Calibration does not match the ONNX graph manifest")
    if source.get("graph_manifest_sha256") != graph_hash:
        raise ValueError("Uploaded source models do not match the ONNX graph manifest")
    if source.get("calibration_manifest_sha256") != calibration_hash:
        raise ValueError("Source manifest does not match the calibration manifest")
    if source.get("compile_options") != _compile_options():
        raise ValueError("Source manifest has unexpected QNN DLC compile options")

    spec = _graph_specs(graph_manifest).get(args.graph)
    graph_calibration = calibration["graphs"].get(args.graph)
    source_record = source["graphs"].get(args.graph)
    if spec is None or graph_calibration is None or source_record is None:
        raise ValueError(f"Graph is missing from a manifest: {args.graph}")
    if source_record.get("onnx_sha256") != graph_calibration.get("onnx_sha256"):
        raise ValueError("Uploaded source ONNX checksum does not match calibration")
    source_model_id = source_record.get("source_model_id")
    if not isinstance(source_model_id, str) or not source_model_id:
        raise ValueError(f"No uploaded original ONNX model ID for {args.graph}")
    device = source.get("target_device")
    if not isinstance(device, dict) or not device.get("name"):
        raise ValueError("Source manifest has no target device")

    sample = next(
        (
            item
            for item in graph_calibration["samples"]
            if item["source_id"] == args.source_id and item["frame"] == args.frame
        ),
        None,
    )
    if sample is None:
        raise ValueError(
            f"No calibration sample for {args.graph} "
            f"source={args.source_id} frame={args.frame}"
        )
    config = {
        "format": FORMAT,
        "graph": args.graph,
        "graph_manifest_sha256": graph_hash,
        "calibration_manifest_sha256": calibration_hash,
        "onnx_sha256": source_record["onnx_sha256"],
        "source_model_id": source_model_id,
        "target_device": device,
        "compile_options": source["compile_options"],
        "sample": sample,
    }
    return config, spec, sample


def _compiled_model(
    config: dict[str, Any], state_path: Path, input_specs: dict[str, Any]
) -> tuple[str, Any]:
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {"config": config}
    )
    if state.get("config") != config:
        raise ValueError(
            "Float-control configuration changed; use a new output directory"
        )
    if "compile_job_id" not in state:
        job = hub.submit_compile_job(
            model=hub.get_model(config["source_model_id"]),
            input_specs=input_specs,
            device=hub.Device(**config["target_device"]),
            name=f"moshi-{config['graph']}-unquantized-control",
            options=config["compile_options"],
        )
        state["compile_job_id"] = job.job_id
        _write_json(state_path, state)
        print(f"Unquantized compile submitted: {job.url}", flush=True)
    else:
        job = hub.get_job(state["compile_job_id"])
        print(f"Unquantized compile resumed: {job.url}", flush=True)
    job.wait()
    status = job.get_status()
    if not status.success:
        raise RuntimeError(f"Unquantized compile did not succeed: {job.url}")
    model = job.get_target_model()
    if model is None:
        raise RuntimeError(f"Unquantized compile returned no model: {job.url}")
    if state.get("compiled_model_id", model.model_id) != model.model_id:
        raise ValueError("Recorded compiled model ID changed")
    state["compiled_model_id"] = model.model_id
    _write_json(state_path, state)
    return model.model_id, job


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-dir", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--source-quantization-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--graph", default="temporal_layers_30_31")
    parser.add_argument("--source-id", default="clean-000")
    parser.add_argument("--frame", type=int, default=0)
    args = parser.parse_args()
    if not args.graph.startswith("temporal_layers_"):
        raise SystemExit("This control probes a Temporal shard with a position input")

    config, spec, sample = _source_and_sample(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir / "compile.json"
    report_path = args.output_dir / "report.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("config") != config:
            raise ValueError(
                "Float-control configuration changed; use a new output directory"
            )
        if report_path.exists():
            print(report_path.read_text(), end="", flush=True)
            print(
                "Existing float-control report reused; no cloud job submitted",
                flush=True,
            )
            return

    with np.load(args.calibration_dir / sample["file"]) as arrays:
        feed = {name: np.array(arrays[name], copy=True) for name in spec["input_names"]}
    position_value = np.asarray(feed["position"])
    if position_value.size != 1 or int(position_value.item()) != args.frame:
        raise ValueError("Captured position does not match the requested frame")
    print(
        f"Float control: graph={args.graph} source_model={config['source_model_id']} "
        f"sample={args.source_id}/frame_{args.frame} "
        f"options={config['compile_options']}",
        flush=True,
    )
    input_specs = {
        name: (tuple(value.shape), str(value.dtype)) for name, value in feed.items()
    }
    compiled_model_id, job = _compiled_model(config, state_path, input_specs)
    input_order = list(job.get_target_shapes())
    if set(input_order) != set(spec["input_names"]):
        raise ValueError(f"Compiled input names changed: {input_order}")

    session = _session(args.onnx_dir / spec["onnx"])
    expected = session.run(spec["output_names"], feed)
    actual = _cloud_batch(
        model_id=compiled_model_id,
        device=config["target_device"],
        samples=[feed],
        input_order=input_order,
        output_names=spec["output_names"],
        stem=args.output_dir / "float_probe",
        retry_failed=False,
    )[0]
    position = int(np.asarray(feed["position"]).item())
    outputs = {}
    for index, (name, received, reference) in enumerate(
        zip(spec["output_names"], actual, expected, strict=True)
    ):
        metrics = _probe_metrics(received, reference)
        metrics["actual_nonzero"] = int(np.count_nonzero(received))
        metrics["reference_nonzero"] = int(np.count_nonzero(reference))
        if index > 0 and received.ndim >= 3:
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
        "sample": sample,
        "position": position,
        "source_model_id": config["source_model_id"],
        "compiled_model_id": compiled_model_id,
        "compile_job_id": job.job_id,
        "all_outputs_finite": all(value["finite"] for value in outputs.values()),
        "outputs": outputs,
    }
    _write_json(report_path, report)
    print(json.dumps(report, indent=2), flush=True)
    print(
        "Unquantized control finished; inspect numerical errors before using the model",
        flush=True,
    )


if __name__ == "__main__":
    main()

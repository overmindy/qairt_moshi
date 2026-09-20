"""Verify a quantized Moshi LM graph set with chained AI Hub inference."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import qai_hub as hub
import torch

GRAPH_SET_FORMAT = "moshi-lm-onnx-graph-set-v1"
CALIBRATION_FORMAT = "moshi-lm-graph-calibration-v1"
QUANTIZATION_FORMAT = "moshi-lm-ai-hub-quantization-v1"
RUN_FORMAT = "moshi-lm-ai-hub-chained-validation-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def _metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    if actual.shape != expected.shape:
        raise ValueError(f"Shape mismatch: {actual.shape} != {expected.shape}")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Metric input contains NaN or Inf")
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    rmse = math.sqrt(float(np.mean(delta * delta)))
    reference_rms = math.sqrt(float(np.mean(expected.astype(np.float64) ** 2)))
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "rmse": rmse,
        "relative_rms": rmse / max(reference_rms, 1e-12),
    }


def _load_trace_tensor(trace_dir: Path, name: str) -> np.ndarray:
    return (
        torch.load(trace_dir / f"{name}.pt", map_location="cpu", weights_only=True)
        .float()
        .numpy()
    )


def _qnn_input(value: np.ndarray) -> np.ndarray:
    """Match DLC inputs produced with ``--truncate_64bit_io``."""
    value = np.asarray(value)
    if value.dtype != np.int64:
        return value
    limits = np.iinfo(np.int32)
    if value.size and (value.min() < limits.min or value.max() > limits.max):
        raise ValueError("int64 input cannot be represented by QNN int32 I/O")
    return value.astype(np.int32)


def _batch_signature(
    model_id: str,
    device: dict[str, str],
    output_names: list[str],
    samples: list[dict[str, np.ndarray]],
) -> str:
    digest = hashlib.sha256()
    digest.update(model_id.encode())
    digest.update(json.dumps(device, sort_keys=True).encode())
    digest.update(json.dumps(output_names).encode())
    for sample in samples:
        for name, value in sample.items():
            contiguous = np.ascontiguousarray(value)
            digest.update(f"{name}:{contiguous.shape}:{contiguous.dtype}".encode())
            digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _order_samples(
    samples: list[dict[str, np.ndarray]], input_order: list[str]
) -> list[dict[str, np.ndarray]]:
    expected = set(input_order)
    ordered = []
    for sample in samples:
        if set(sample) != expected:
            raise ValueError(
                f"Inference inputs {list(sample)} do not match compiled inputs "
                f"{input_order}"
            )
        ordered.append({name: sample[name] for name in input_order})
    return ordered


def _load_output_archive(
    path: Path, sample_count: int, output_count: int
) -> list[list[np.ndarray]]:
    with np.load(path) as archive:
        return [
            [
                archive[f"sample_{sample}_output_{output}"].copy()
                for output in range(output_count)
            ]
            for sample in range(sample_count)
        ]


def _cloud_batch(
    *,
    model_id: str,
    device: dict[str, str],
    samples: list[dict[str, np.ndarray]],
    input_order: list[str],
    output_names: list[str],
    stem: Path,
    retry_failed: bool,
    reuse_stem: Path | None = None,
) -> list[list[np.ndarray]]:
    if not samples:
        raise ValueError("Cloud inference batch cannot be empty")
    samples = _order_samples(samples, input_order)
    input_names = input_order
    converted = [
        {name: _qnn_input(value) for name, value in sample.items()}
        for sample in samples
    ]
    signature = _batch_signature(model_id, device, output_names, converted)
    record_path = stem.with_suffix(".json")
    archive_path = stem.with_suffix(".npz")
    if reuse_stem is not None and not record_path.exists() and not archive_path.exists():
        old_record_path = reuse_stem.with_suffix(".json")
        old_archive_path = reuse_stem.with_suffix(".npz")
        if old_record_path.is_file() and old_archive_path.is_file():
            old_record = json.loads(old_record_path.read_text())
            if old_record.get("signature") == signature and old_record.get("job_id"):
                old_job = hub.get_job(old_record["job_id"])
                if not old_job.get_status().success:
                    raise RuntimeError(
                        f"Matching cached inference job is not successful: {old_job.url}"
                    )
                expected_archive_names = {
                    f"sample_{sample}_output_{output}"
                    for sample in range(len(converted))
                    for output in range(len(output_names))
                }
                with np.load(old_archive_path) as old_archive:
                    if set(old_archive.files) != expected_archive_names:
                        raise ValueError(
                            f"Cached output archive has unexpected keys: {old_archive_path}"
                        )
                _write_json(
                    record_path,
                    {**old_record, "reused_from": str(old_record_path)},
                )
                try:
                    os.link(old_archive_path, archive_path)
                except OSError:
                    temporary_archive = archive_path.with_suffix(".reuse.tmp.npz")
                    shutil.copy2(old_archive_path, temporary_archive)
                    temporary_archive.replace(archive_path)
                print(f"[reuse] {stem.name}: verified job={old_job.job_id}", flush=True)
            else:
                print(f"[reuse] {stem.name}: input/model signature changed", flush=True)
    record: dict[str, Any] = {}
    if record_path.exists():
        record = json.loads(record_path.read_text())
        if record.get("signature") != signature:
            if not retry_failed or not record.get("job_id"):
                raise ValueError(
                    f"Inputs/model changed; use a new run directory: {stem}"
                )
            previous_job = hub.get_job(record["job_id"])
            if not previous_job.get_status().failure:
                raise ValueError(
                    f"Inputs/model changed but the recorded job is not failed: "
                    f"{previous_job.url}"
                )
            failed_job_ids = [
                *record.get("failed_job_ids", []),
                previous_job.job_id,
            ]
            record = {"failed_job_ids": list(dict.fromkeys(failed_job_ids))}
    if archive_path.exists():
        return _load_output_archive(archive_path, len(converted), len(output_names))

    job = None
    if record.get("job_id"):
        job = hub.get_job(record["job_id"])
        status = job.get_status()
        if status.failure:
            if not retry_failed:
                raise RuntimeError(
                    f"Recorded inference job failed: {job.url}; "
                    "pass --retry-failed to replace it"
                )
            record.setdefault("failed_job_ids", []).append(job.job_id)
            record.pop("job_id", None)
            job = None
    if job is None:
        job = hub.submit_inference_job(
            model=hub.get_model(model_id),
            device=hub.Device(**device),
            inputs={
                name: [sample[name] for sample in converted] for name in input_names
            },
            name=f"moshi-quantized-check-{stem.name}",
        )
        record.update(
            {
                "signature": signature,
                "job_id": job.job_id,
                "target_model_id": model_id,
                "samples": len(converted),
                "output_names": output_names,
            }
        )
        _write_json(record_path, record)
    print(f"{stem.name}: job={job.job_id} samples={len(converted)}", flush=True)
    job.wait()
    status = job.get_status()
    if not status.success:
        raise RuntimeError(f"Inference job did not succeed: {job.url}")
    downloaded = job.download_output_data()
    if downloaded is None:
        raise RuntimeError(f"Inference job returned no output data: {job.url}")
    named = set(output_names)
    numbered = {f"output_{index}" for index in range(len(output_names))}
    if set(downloaded) == named:
        keys = output_names
    elif set(downloaded) == numbered:
        keys = [f"output_{index}" for index in range(len(output_names))]
    else:
        raise ValueError(f"Unexpected cloud outputs: {list(downloaded)}")
    if any(len(downloaded[key]) != len(converted) for key in keys):
        raise ValueError("Cloud output sample count does not match the input batch")
    outputs = [
        [np.asarray(downloaded[key][sample]) for key in keys]
        for sample in range(len(converted))
    ]
    values = {
        f"sample_{sample}_output_{output}": value
        for sample, sample_outputs in enumerate(outputs)
        for output, value in enumerate(sample_outputs)
    }
    temporary = archive_path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **values)
    temporary.replace(archive_path)
    return outputs


def _select_sources(
    calibration: dict[str, Any], requested: list[str] | None
) -> list[dict[str, Any]]:
    sources = calibration.get("sources", [])
    by_id = {source["id"]: source for source in sources}
    if requested:
        missing = sorted(set(requested) - set(by_id))
        if missing:
            raise ValueError(f"Unknown calibration source IDs: {missing}")
        selected = [by_id[source_id] for source_id in requested]
    else:
        selected = []
        for kind in ("clean", "overlap"):
            match = next(
                (source for source in sources if kind in source["id"].lower()), None
            )
            if match is None:
                raise ValueError(f"No {kind} source found; pass --source-id explicitly")
            selected.append(match)
    if len({source["id"] for source in selected}) != len(selected):
        raise ValueError("Source IDs must be unique")
    return selected


def _target_id(quantization: dict[str, Any], graph_name: str) -> str:
    record = quantization["graphs"].get(graph_name)
    if record is None:
        raise ValueError(f"Quantization manifest is missing graph {graph_name}")
    model_id = record.get("compiled_model_id")
    if not model_id:
        raise ValueError(f"Graph {graph_name} has no compiled model ID")
    return model_id


def _compiled_input_order(quantization: dict[str, Any], graph_name: str) -> list[str]:
    record = quantization["graphs"].get(graph_name)
    if record is None:
        raise ValueError(f"Quantization manifest is missing graph {graph_name}")
    job_id = record.get("compile_job_id")
    if not job_id:
        raise ValueError(f"Graph {graph_name} has no compile job ID")
    job = hub.get_job(job_id)
    status = job.get_status()
    if not status.success:
        raise ValueError(f"Compile job is not successful for {graph_name}: {job.url}")
    input_order = list(job.get_target_shapes())
    if set(input_order) != set(record["input_names"]):
        raise ValueError(
            f"Compiled inputs for {graph_name} changed: {input_order} != "
            f"{record['input_names']}"
        )
    return input_order


def _cache_report(
    actual: np.ndarray,
    expected: np.ndarray,
    previous: np.ndarray,
    slot: int,
) -> dict[str, Any]:
    if actual.shape != expected.shape or actual.shape != previous.shape:
        raise ValueError("Cache shape changed")
    if actual.ndim < 3 or slot >= actual.shape[2]:
        raise ValueError(f"Invalid cache slot {slot} for shape {actual.shape}")
    preserved_max_abs = 0.0
    for lower, upper in ((0, slot), (slot + 1, actual.shape[2])):
        if lower < upper:
            delta = actual[:, :, lower:upper].astype(np.float64) - previous[
                :, :, lower:upper
            ].astype(np.float64)
            preserved_max_abs = max(preserved_max_abs, float(np.max(np.abs(delta))))
    return {
        "written": _metrics(
            actual[:, :, slot : slot + 1], expected[:, :, slot : slot + 1]
        ),
        "preserved_max_abs": preserved_max_abs,
    }


def _final_report(
    *,
    text_logits: np.ndarray,
    audio_logits: np.ndarray,
    reference_text_logits: np.ndarray,
    reference_audio_logits: np.ndarray,
) -> dict[str, Any]:
    text_tokens = np.argmax(text_logits, axis=-1)
    reference_text_tokens = np.argmax(reference_text_logits, axis=-1)
    audio_tokens = np.argmax(audio_logits, axis=-1)[..., 0]
    audio_tokens = np.transpose(audio_tokens, (0, 2, 1))
    reference_audio_tokens = np.argmax(reference_audio_logits, axis=-1)[..., 0]
    reference_audio_tokens = np.transpose(reference_audio_tokens, (0, 2, 1))
    return {
        "text_logits": _metrics(text_logits, reference_text_logits),
        "audio_logits": _metrics(audio_logits, reference_audio_logits),
        "text_token_agreement": float(np.mean(text_tokens == reference_text_tokens)),
        "audio_token_agreement": float(np.mean(audio_tokens == reference_audio_tokens)),
    }


def _temporal_error_summary(
    stage_metrics: list[dict[str, Any]], source_id: str, frames: int
) -> list[dict[str, float | int]]:
    summary = []
    for frame in range(frames):
        records = [
            record
            for record in stage_metrics
            if record["kind"] == "temporal"
            and record["source_id"] == source_id
            and record["frame"] == frame
        ]
        cache_entries = [
            cache for record in records for cache in record.get("cache", {}).values()
        ]
        summary.append(
            {
                "frame": frame,
                "max_hidden_relative_rms": max(
                    (record["hidden"]["relative_rms"] for record in records),
                    default=0.0,
                ),
                "max_written_cache_relative_rms": max(
                    (cache["written"]["relative_rms"] for cache in cache_entries),
                    default=0.0,
                ),
                "max_preserved_cache_abs": max(
                    (cache["preserved_max_abs"] for cache in cache_entries),
                    default=0.0,
                ),
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-dir", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--quantization-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--source-id", action="append")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument(
        "--reuse-stage-dir",
        type=Path,
        help="Reuse successful inference archives from a prior chain only when "
        "the model, device, outputs, and every input tensor match exactly",
    )
    args = parser.parse_args()
    if args.frames < 2:
        raise SystemExit("--frames must be at least two to exercise KV cache reuse")
    if args.reuse_stage_dir is not None:
        if args.reuse_stage_dir.resolve() == args.output_dir.resolve():
            raise SystemExit("--reuse-stage-dir must differ from --output-dir")
        if not args.reuse_stage_dir.is_dir():
            raise SystemExit(f"Missing --reuse-stage-dir: {args.reuse_stage_dir}")

    graph_path = args.onnx_dir / "manifest.json"
    calibration_path = args.calibration_dir / "manifest.json"
    quantization_path = args.quantization_dir / "quantization_manifest.json"
    graph = json.loads(graph_path.read_text())
    calibration = json.loads(calibration_path.read_text())
    quantization = json.loads(quantization_path.read_text())
    if graph.get("format") != GRAPH_SET_FORMAT:
        raise SystemExit(f"Unsupported graph manifest: {graph.get('format')}")
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
    device = quantization.get("target_device")
    if not isinstance(device, dict) or not device.get("name"):
        raise SystemExit("Quantization manifest has no target device")
    compiled_input_orders: dict[str, list[str]] = {}

    def input_order(graph_name: str) -> list[str]:
        if graph_name not in compiled_input_orders:
            compiled_input_orders[graph_name] = _compiled_input_order(
                quantization, graph_name
            )
        return compiled_input_orders[graph_name]

    selected_sources = _select_sources(calibration, args.source_id)
    sources: list[dict[str, Any]] = []
    for source in selected_sources:
        trace_dir = Path(source["trace_dir"])
        sequence = torch.load(
            trace_dir / "temporal_sequence.pt",
            map_location="cpu",
            weights_only=True,
        ).numpy()
        if sequence.shape[-1] < args.frames:
            raise SystemExit(
                f"Source {source['id']} has only {sequence.shape[-1]} frames"
            )
        sources.append(
            {
                "id": source["id"],
                "trace_dir": trace_dir,
                "sequence": sequence[..., : args.frames],
                "bf16_text_logits": _load_trace_tensor(trace_dir, "text_logits")[
                    ..., : args.frames, :
                ],
                "bf16_audio_logits": _load_trace_tensor(trace_dir, "depformer_logits")[
                    :, : args.frames
                ],
            }
        )

    config = {
        "format": RUN_FORMAT,
        "graph_manifest_sha256": _sha256(graph_path),
        "calibration_manifest_sha256": _sha256(calibration_path),
        "quantization_manifest_sha256": _sha256(quantization_path),
        "frames": args.frames,
        "sources": [
            {"id": source["id"], "trace_dir": str(source["trace_dir"])}
            for source in sources
        ],
        "target_device": device,
    }
    if args.reuse_stage_dir is not None:
        config["reuse_stage_dir"] = str(args.reuse_stage_dir.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "run_manifest.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise SystemExit("Validation configuration changed; use a new --output-dir")
    _write_json(config_path, config)

    def chain_batch(**kwargs: Any) -> list[list[np.ndarray]]:
        stem = kwargs["stem"]
        return _cloud_batch(
            **kwargs,
            reuse_stem=(
                args.reuse_stage_dir / stem.name
                if args.reuse_stage_dir is not None
                else None
            ),
        )

    metrics_path = args.output_dir / "stage_metrics.json"
    stage_metrics: list[dict[str, Any]] = []
    expected_jobs = (
        2
        + len(graph["temporal_shards"]) * args.frames
        + len(graph["depformer"]["steps"])
    )
    print(
        f"Chained validation plan: {len(sources)} sources, {args.frames} frames, "
        f"{expected_jobs} resumable inference jobs",
        flush=True,
    )

    frontend = graph["frontend"]
    frontend_name = "frontend"
    session = _session(args.onnx_dir / frontend["onnx"])
    cpu_hidden: dict[str, list[np.ndarray]] = {source["id"]: [] for source in sources}
    frontend_samples = []
    sample_keys = []
    for source in sources:
        for frame in range(args.frames):
            feed = {"sequence": source["sequence"][..., frame : frame + 1]}
            cpu_hidden[source["id"]].append(
                session.run(frontend["output_names"], feed)[0]
            )
            frontend_samples.append(feed)
            sample_keys.append((source["id"], frame))
    cloud_outputs = chain_batch(
        model_id=_target_id(quantization, frontend_name),
        device=device,
        samples=frontend_samples,
        input_order=input_order(frontend_name),
        output_names=frontend["output_names"],
        stem=args.output_dir / "000_frontend",
        retry_failed=args.retry_failed,
    )
    cloud_hidden: dict[str, list[np.ndarray]] = {
        source["id"]: [np.empty(0)] * args.frames for source in sources
    }
    for (source_id, frame), received in zip(sample_keys, cloud_outputs, strict=True):
        cloud_hidden[source_id][frame] = received[0]
        stage_metrics.append(
            {
                "kind": "frontend",
                "graph": frontend_name,
                "source_id": source_id,
                "frame": frame,
                "hidden": _metrics(received[0], cpu_hidden[source_id][frame]),
            }
        )
    _write_json(metrics_path, stage_metrics)
    del session, frontend_samples, cloud_outputs
    gc.collect()

    for shard_index, shard in enumerate(graph["temporal_shards"], start=1):
        graph_name = Path(shard["onnx"]).stem
        session = _session(args.onnx_dir / shard["onnx"])
        cpu_caches = {
            source["id"]: [
                np.zeros(shape, np.float32) for shape in shard["cache_shapes"]
            ]
            for source in sources
        }
        cloud_caches = {
            source["id"]: [
                np.zeros(shape, np.float32) for shape in shard["cache_shapes"]
            ]
            for source in sources
        }
        next_cpu = {source["id"]: [] for source in sources}
        next_cloud = {source["id"]: [] for source in sources}
        for frame in range(args.frames):
            samples = []
            expected_by_source = {}
            previous_by_source = {}
            for source in sources:
                source_id = source["id"]
                position = np.array([frame], dtype=np.int64)
                cpu_feed = dict(
                    zip(
                        shard["input_names"],
                        [
                            cpu_hidden[source_id][frame],
                            position,
                            *cpu_caches[source_id],
                        ],
                        strict=True,
                    )
                )
                expected = session.run(shard["output_names"], cpu_feed)
                expected_by_source[source_id] = expected
                next_cpu[source_id].append(expected[0])
                cpu_caches[source_id] = expected[1:]
                previous_by_source[source_id] = cloud_caches[source_id]
                samples.append(
                    dict(
                        zip(
                            shard["input_names"],
                            [
                                cloud_hidden[source_id][frame],
                                position,
                                *cloud_caches[source_id],
                            ],
                            strict=True,
                        )
                    )
                )
            received_batch = chain_batch(
                model_id=_target_id(quantization, graph_name),
                device=device,
                samples=samples,
                input_order=input_order(graph_name),
                output_names=shard["output_names"],
                stem=args.output_dir
                / f"{shard_index:03d}_{graph_name}_frame_{frame:04d}",
                retry_failed=args.retry_failed,
            )
            for source, received in zip(sources, received_batch, strict=True):
                source_id = source["id"]
                expected = expected_by_source[source_id]
                cache_metrics = {
                    name: _cache_report(actual, reference, previous, frame)
                    for name, actual, reference, previous in zip(
                        shard["output_names"][1:],
                        received[1:],
                        expected[1:],
                        previous_by_source[source_id],
                        strict=True,
                    )
                }
                stage_metrics.append(
                    {
                        "kind": "temporal",
                        "graph": graph_name,
                        "source_id": source_id,
                        "frame": frame,
                        "hidden": _metrics(received[0], expected[0]),
                        "cache": cache_metrics,
                    }
                )
                next_cloud[source_id].append(received[0])
                cloud_caches[source_id] = received[1:]
            _write_json(metrics_path, stage_metrics)
        cpu_hidden = next_cpu
        cloud_hidden = next_cloud
        del session, cpu_caches, cloud_caches
        gc.collect()

    head = graph["head"]
    head_name = "head"
    session = _session(args.onnx_dir / head["onnx"])
    head_samples = []
    sample_keys = []
    cpu_temporal = {source["id"]: [np.empty(0)] * args.frames for source in sources}
    cpu_text_frames = {source["id"]: [np.empty(0)] * args.frames for source in sources}
    for source in sources:
        source_id = source["id"]
        for frame in range(args.frames):
            expected = session.run(
                head["output_names"], {"hidden": cpu_hidden[source_id][frame]}
            )
            cpu_temporal[source_id][frame], cpu_text_frames[source_id][frame] = expected
            head_samples.append({"hidden": cloud_hidden[source_id][frame]})
            sample_keys.append((source_id, frame))
    received_batch = chain_batch(
        model_id=_target_id(quantization, head_name),
        device=device,
        samples=head_samples,
        input_order=input_order(head_name),
        output_names=head["output_names"],
        stem=args.output_dir / "017_head",
        retry_failed=args.retry_failed,
    )
    cloud_temporal = {source["id"]: [np.empty(0)] * args.frames for source in sources}
    cloud_text_frames = {
        source["id"]: [np.empty(0)] * args.frames for source in sources
    }
    for (source_id, frame), received in zip(sample_keys, received_batch, strict=True):
        cloud_temporal[source_id][frame], cloud_text_frames[source_id][frame] = received
        stage_metrics.append(
            {
                "kind": "head",
                "graph": head_name,
                "source_id": source_id,
                "frame": frame,
                "temporal": _metrics(received[0], cpu_temporal[source_id][frame]),
                "text_logits": _metrics(received[1], cpu_text_frames[source_id][frame]),
            }
        )
    _write_json(metrics_path, stage_metrics)
    del session, head_samples, received_batch
    gc.collect()

    cpu_previous = {
        source["id"]: [
            np.argmax(value, axis=-1)[:, 0, 0].astype(np.int64)
            for value in cpu_text_frames[source["id"]]
        ]
        for source in sources
    }
    cloud_previous = {
        source["id"]: [
            np.argmax(value, axis=-1)[:, 0, 0].astype(np.int64)
            for value in cloud_text_frames[source["id"]]
        ]
        for source in sources
    }
    cpu_cache_frames = None
    cloud_cache_frames = None
    cpu_audio_logits = {
        source["id"]: [[] for _ in range(args.frames)] for source in sources
    }
    cloud_audio_logits = {
        source["id"]: [[] for _ in range(args.frames)] for source in sources
    }
    for step_index, step in enumerate(graph["depformer"]["steps"]):
        graph_name = Path(step["onnx"]).stem
        session = _session(args.onnx_dir / step["onnx"])
        if cpu_cache_frames is None:
            cpu_cache_frames = {
                source["id"]: [
                    [np.zeros(shape, np.float32) for shape in step["cache_shapes"]]
                    for _ in range(args.frames)
                ]
                for source in sources
            }
            cloud_cache_frames = {
                source["id"]: [
                    [np.zeros(shape, np.float32) for shape in step["cache_shapes"]]
                    for _ in range(args.frames)
                ]
                for source in sources
            }
        assert cloud_cache_frames is not None
        samples = []
        sample_keys = []
        expected_by_key = {}
        previous_by_key = {}
        next_cpu_cache = {source["id"]: [None] * args.frames for source in sources}
        next_cloud_cache = {source["id"]: [None] * args.frames for source in sources}
        next_cpu_token = {source["id"]: [None] * args.frames for source in sources}
        next_cloud_token = {source["id"]: [None] * args.frames for source in sources}
        for source in sources:
            source_id = source["id"]
            for frame in range(args.frames):
                cpu_feed = dict(
                    zip(
                        step["input_names"],
                        [
                            cpu_previous[source_id][frame],
                            cpu_temporal[source_id][frame],
                            *cpu_cache_frames[source_id][frame],
                        ],
                        strict=True,
                    )
                )
                expected = session.run(step["output_names"], cpu_feed)
                key = (source_id, frame)
                expected_by_key[key] = expected
                next_cpu_token[source_id][frame] = expected[0]
                next_cpu_cache[source_id][frame] = expected[2:]
                cpu_audio_logits[source_id][frame].append(expected[1])
                previous_by_key[key] = cloud_cache_frames[source_id][frame]
                samples.append(
                    dict(
                        zip(
                            step["input_names"],
                            [
                                cloud_previous[source_id][frame],
                                cloud_temporal[source_id][frame],
                                *cloud_cache_frames[source_id][frame],
                            ],
                            strict=True,
                        )
                    )
                )
                sample_keys.append(key)
        received_batch = chain_batch(
            model_id=_target_id(quantization, graph_name),
            device=device,
            samples=samples,
            input_order=input_order(graph_name),
            output_names=step["output_names"],
            stem=args.output_dir / f"{18 + step_index:03d}_{graph_name}",
            retry_failed=args.retry_failed,
        )
        for key, received in zip(sample_keys, received_batch, strict=True):
            source_id, frame = key
            expected = expected_by_key[key]
            cache_metrics = {
                name: _cache_report(actual, reference, previous, step["codebook"])
                for name, actual, reference, previous in zip(
                    step["output_names"][2:],
                    received[2:],
                    expected[2:],
                    previous_by_key[key],
                    strict=True,
                )
            }
            stage_metrics.append(
                {
                    "kind": "depformer",
                    "graph": graph_name,
                    "source_id": source_id,
                    "frame": frame,
                    "codebook": step["codebook"],
                    "token_match": bool(np.array_equal(received[0], expected[0])),
                    "audio_logits": _metrics(received[1], expected[1]),
                    "cache": cache_metrics,
                }
            )
            next_cloud_token[source_id][frame] = received[0]
            next_cloud_cache[source_id][frame] = received[2:]
            cloud_audio_logits[source_id][frame].append(received[1])
        cpu_previous = next_cpu_token
        cloud_previous = next_cloud_token
        cpu_cache_frames = next_cpu_cache
        cloud_cache_frames = next_cloud_cache
        _write_json(metrics_path, stage_metrics)
        del session, samples, received_batch
        gc.collect()

    report: dict[str, Any] = {
        "format": RUN_FORMAT,
        "frames": args.frames,
        "sources": {},
    }
    quantization_passed = True
    bf16_passed = True
    for source in sources:
        source_id = source["id"]
        cpu_text = np.concatenate(cpu_text_frames[source_id], axis=2)
        cloud_text = np.concatenate(cloud_text_frames[source_id], axis=2)
        cpu_audio = np.stack(
            [np.concatenate(values, axis=1) for values in cpu_audio_logits[source_id]],
            axis=1,
        )
        cloud_audio = np.stack(
            [
                np.concatenate(values, axis=1)
                for values in cloud_audio_logits[source_id]
            ],
            axis=1,
        )
        source_report = {
            "fp32_onnx_vs_bf16": _final_report(
                text_logits=cpu_text,
                audio_logits=cpu_audio,
                reference_text_logits=source["bf16_text_logits"],
                reference_audio_logits=source["bf16_audio_logits"],
            ),
            "quantized_vs_fp32_onnx": _final_report(
                text_logits=cloud_text,
                audio_logits=cloud_audio,
                reference_text_logits=cpu_text,
                reference_audio_logits=cpu_audio,
            ),
            "quantized_vs_bf16": _final_report(
                text_logits=cloud_text,
                audio_logits=cloud_audio,
                reference_text_logits=source["bf16_text_logits"],
                reference_audio_logits=source["bf16_audio_logits"],
            ),
        }
        quantized_tokens = source_report["quantized_vs_fp32_onnx"]
        source_quantization_passed = (
            quantized_tokens["text_token_agreement"] == 1.0
            and quantized_tokens["audio_token_agreement"] == 1.0
        )
        bf16_tokens = source_report["quantized_vs_bf16"]
        source_bf16_passed = (
            bf16_tokens["text_token_agreement"] == 1.0
            and bf16_tokens["audio_token_agreement"] == 1.0
        )
        source_report["temporal_error_by_frame"] = _temporal_error_summary(
            stage_metrics, source_id, args.frames
        )
        source_report["quantization_token_parity_pass"] = source_quantization_passed
        source_report["bf16_end_to_end_token_parity_pass"] = source_bf16_passed
        quantization_passed = quantization_passed and source_quantization_passed
        bf16_passed = bf16_passed and source_bf16_passed
        report["sources"][source_id] = source_report
    report["quantization_token_parity_pass"] = quantization_passed
    report["bf16_end_to_end_token_parity_pass"] = bf16_passed
    _write_json(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    if not quantization_passed:
        raise RuntimeError(
            "Quantized-versus-FP32 chained token parity failed; inspect report.json and "
            "stage_metrics.json before linking"
        )
    print("Quantized-versus-FP32 clean/overlap chained token parity: PASS", flush=True)
    if not bf16_passed:
        print(
            "BF16 end-to-end token parity: FAIL; this is reported separately so an "
            "existing FP32/BF16 baseline mismatch is not attributed to quantization",
            flush=True,
        )


if __name__ == "__main__":
    main()

"""Batch-quantize a captured Moshi LM ONNX graph set on Qualcomm AI Hub.

The calibration directory must be produced by ``moshi_run_lm_onnx.py`` from
real sequential execution.  Quantization is submitted for every graph in
parallel; compilation is optional and has no default device.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qai_hub_models import Precision, TargetRuntime

GRAPH_SET_FORMAT = "moshi-lm-onnx-graph-set-v1"
CALIBRATION_FORMAT = "moshi-lm-graph-calibration-v1"
RESULT_FORMAT = "moshi-lm-ai-hub-quantization-v1"
PREFLIGHT_RECEIPT_VERSION = 1
COMPILE_RUNTIME = TargetRuntime.QNN_DLC


def _compile_options() -> str:
    """Return the AI Hub compile flag for the Moshi QNN DLC stage.

    A QNN context binary is not a compile target in current AI Hub clients.
    The supported flow is compile(qnn_dlc), followed by a separate link job
    when a context binary is needed.
    """
    options = COMPILE_RUNTIME.aihub_target_runtime_flag
    if options is None:
        raise RuntimeError(f"AI Hub cannot compile {COMPILE_RUNTIME.value}")
    return options


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _file_stamp(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _preflight_file_stamps(
    onnx_dir: Path,
    calibration_dir: Path,
    graph_specs: list[tuple[str, str, dict[str, Any]]],
    calibration: dict[str, Any],
) -> dict[str, dict[str, dict[str, int]]]:
    onnx_files = {
        spec["onnx"]: _file_stamp(onnx_dir / spec["onnx"]) for _, _, spec in graph_specs
    }
    calibration_files = {
        sample["file"]: _file_stamp(calibration_dir / sample["file"])
        for graph in calibration["graphs"].values()
        for sample in graph.get("samples", [])
    }
    return {"onnx": onnx_files, "calibration": calibration_files}


def _build_preflight_receipt(
    onnx_dir: Path,
    calibration_dir: Path,
    graph_specs: list[tuple[str, str, dict[str, Any]]],
    calibration: dict[str, Any],
    planned: dict[str, Any],
    minimum_sources: int,
    validation: str,
) -> dict[str, Any]:
    return {
        "version": PREFLIGHT_RECEIPT_VERSION,
        "graph_manifest_sha256": planned["graph_manifest_sha256"],
        "calibration_manifest_sha256": planned["calibration_manifest_sha256"],
        "minimum_sources": minimum_sources,
        "validation": validation,
        "files": _preflight_file_stamps(
            onnx_dir, calibration_dir, graph_specs, calibration
        ),
    }


def _preflight_receipt_matches(
    receipt: dict[str, Any] | None,
    onnx_dir: Path,
    calibration_dir: Path,
    graph_specs: list[tuple[str, str, dict[str, Any]]],
    calibration: dict[str, Any],
    planned: dict[str, Any],
    minimum_sources: int,
) -> tuple[bool, str]:
    if receipt is None:
        return False, "no cached receipt"
    expected = {
        "version": PREFLIGHT_RECEIPT_VERSION,
        "graph_manifest_sha256": planned["graph_manifest_sha256"],
        "calibration_manifest_sha256": planned["calibration_manifest_sha256"],
        "minimum_sources": minimum_sources,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            return False, f"cached {key} changed"
    try:
        current_files = _preflight_file_stamps(
            onnx_dir, calibration_dir, graph_specs, calibration
        )
    except FileNotFoundError as error:
        return False, f"file disappeared: {error.filename}"
    if receipt.get("files") != current_files:
        return False, "an ONNX or calibration file size/mtime changed"
    return True, "manifest hashes and file stamps unchanged"


def _graph_specs(manifest: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    specs = [("frontend", "frontend", manifest["frontend"])]
    specs.extend(
        (Path(item["onnx"]).stem, "temporal", item)
        for item in manifest["temporal_shards"]
    )
    specs.append(("head", "head", manifest["head"]))
    specs.extend(
        (Path(item["onnx"]).stem, "depformer", item)
        for item in manifest["depformer"]["steps"]
    )
    names = [name for name, _, _ in specs]
    if len(names) != len(set(names)):
        raise ValueError(f"Graph names are not unique: {names}")
    return specs


def _precision_for_graph(requested: str, kind: str) -> Precision:
    if requested == "auto":
        return (
            Precision.w8a16_mixed_fp16
            if kind in {"temporal", "depformer"}
            else Precision.w8a16
        )
    precision = Precision.parse(requested)
    if not precision.activations_type or not precision.weights_type:
        raise ValueError(f"{requested!r} is not a quantized W/A precision")
    return precision


def _quantize_options(precision: Precision, litemp_percentage: float) -> str:
    return precision.get_hub_quantize_options(
        litemp_percentage if precision.override_type is not None else None
    )


def _walk_dicts(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_dicts(child)


def _reusable_source_models(path: Path | None) -> dict[str, str]:
    """Return only source model IDs backed by an explicit ONNX SHA256."""
    if path is None:
        return {}
    value = json.loads(path.read_text())
    reusable: dict[str, str] = {}
    for item in _walk_dicts(value):
        checksum = item.get("onnx_sha256") or item.get("sha256")
        model_id = (
            item.get("source_model_id")
            or item.get("uploaded_source_model_id")
            or item.get("model_id")
        )
        if isinstance(checksum, str) and isinstance(model_id, str):
            previous = reusable.get(checksum)
            if previous is not None and previous != model_id:
                raise ValueError(
                    f"Conflicting source model IDs for ONNX SHA256 {checksum}"
                )
            reusable[checksum] = model_id
    return reusable


def _validate_calibration(
    calibration_dir: Path,
    calibration: dict[str, Any],
    graph_specs: list[tuple[str, str, dict[str, Any]]],
    minimum_sources: int,
    *,
    inspect_arrays: bool = True,
) -> None:
    sources = calibration.get("sources", [])
    if len(sources) < minimum_sources:
        raise ValueError(
            f"Calibration has {len(sources)} sources; at least {minimum_sources} required"
        )
    expected_sources = {item["id"] for item in sources}
    for graph_index, (name, _, spec) in enumerate(graph_specs, start=1):
        graph = calibration["graphs"].get(name)
        if graph is None:
            raise ValueError(f"Calibration is missing graph {name}")
        if graph["onnx"] != spec["onnx"]:
            raise ValueError(f"Calibration ONNX mismatch for {name}")
        if graph["input_names"] != spec["input_names"]:
            raise ValueError(f"Calibration input names mismatch for {name}")
        samples = graph.get("samples", [])
        actual_sources = {item["source_id"] for item in samples}
        if actual_sources != expected_sources:
            raise ValueError(
                f"Calibration graph {name} has sources {sorted(actual_sources)}, "
                f"expected {sorted(expected_sources)}"
            )
        expected_keys = {
            (source["id"], frame)
            for source in sources
            for frame in source["frame_positions"]
        }
        actual_keys = {(item["source_id"], item["frame"]) for item in samples}
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            raise ValueError(
                f"Calibration sample mismatch for {name}: missing={missing}, extra={extra}"
            )
        if inspect_arrays:
            print(
                f"[preflight] calibration {graph_index}/{len(graph_specs)}: "
                f"{name} ({len(samples)} samples)",
                flush=True,
            )
        input_signatures: dict[str, tuple[tuple[int, ...], str]] | None = None
        for sample in samples:
            path = calibration_dir / sample["file"]
            if not path.is_file():
                raise ValueError(f"Missing calibration sample {path}")
            if not inspect_arrays:
                continue
            with np.load(path) as arrays:
                if arrays.files != graph["input_names"]:
                    raise ValueError(
                        f"Calibration inputs in {path} are {arrays.files}, "
                        f"expected {graph['input_names']}"
                    )
                current_signatures = {
                    input_name: (
                        tuple(arrays[input_name].shape),
                        str(arrays[input_name].dtype),
                    )
                    for input_name in graph["input_names"]
                }
                if input_signatures is None:
                    input_signatures = current_signatures
                elif current_signatures != input_signatures:
                    raise ValueError(
                        f"Calibration shape/dtype changed in graph {name}: {path}"
                    )


def _load_calibration_entries(
    calibration_dir: Path, graph: dict[str, Any]
) -> dict[str, list[np.ndarray]]:
    """Load captured feeds without changing the ONNX input dtypes.

    The generic ``make_hub_dataset_entries`` helper converts int64 arrays to
    int32 for common device inputs. Moshi's exported ONNX graphs declare
    sequence, position, and previous_token as int64, so that conversion makes
    AI Hub reject the calibration dataset before quantization starts.
    """
    values: list[list[np.ndarray]] = [[] for _ in graph["input_names"]]
    for sample in graph["samples"]:
        with np.load(calibration_dir / sample["file"]) as arrays:
            for index, name in enumerate(graph["input_names"]):
                values[index].append(np.array(arrays[name], copy=True))
    return dict(zip(graph["input_names"], values, strict=True))


def _input_specs(calibration_dir: Path, graph: dict[str, Any]) -> dict[str, Any]:
    with np.load(calibration_dir / graph["samples"][0]["file"]) as arrays:
        return {
            name: (tuple(arrays[name].shape), str(arrays[name].dtype))
            for name in graph["input_names"]
        }


def _status_summary(status: Any) -> str:
    state = getattr(status, "state", None)
    state_name = getattr(state, "name", None) or getattr(status, "code", None)
    message = getattr(status, "message", None)
    return f"{state_name or 'UNKNOWN'}{f': {message}' if message else ''}"


def _required_target_model(job: Any, stage: str) -> Any:
    status = job.get_status()
    if not status.success:
        raise RuntimeError(
            f"{stage} job did not succeed ({_status_summary(status)}): {job.url}"
        )
    target = job.get_target_model()
    if target is None:
        raise RuntimeError(f"{stage} job produced no target model: {job.url}")
    return target


def _recheck_recorded_target(
    hub: Any,
    record: dict[str, Any],
    *,
    stage: str,
    job_id_key: str,
    model_id_key: str,
) -> tuple[str, Any | None]:
    """Verify a recorded target against live job status before skipping it."""
    if not record.get(model_id_key):
        return "missing", None
    job_id = record.get(job_id_key)
    if not job_id:
        record.pop(model_id_key, None)
        record["status"] = f"{stage}_record_invalid"
        record["error"] = f"Recorded {model_id_key} has no associated {job_id_key}"
        return "retry", None

    job = hub.get_job(job_id)
    status = job.get_status()
    record[f"{stage}_live_status"] = _status_summary(status)
    if status.success:
        target = _required_target_model(job, stage.capitalize())
        record[model_id_key] = target.model_id
        record["status"] = f"{stage}_succeeded"
        record.pop("error", None)
        return "success", job
    if status.failure:
        record.pop(model_id_key, None)
        record.pop(job_id_key, None)
        record["status"] = f"{stage}_failed"
        record["error"] = f"Recorded {stage} job failed: {_status_summary(status)}"
        return "retry", None

    record.pop(model_id_key, None)
    record["status"] = f"{stage}_submitted"
    return "running", job


def _build_result(
    graph_manifest_path: Path,
    calibration_manifest_path: Path,
    requested_precision: str,
    litemp_percentage: float,
    graph_specs: list[tuple[str, str, dict[str, Any]]],
    calibration: dict[str, Any],
) -> dict[str, Any]:
    graphs = {}
    for name, kind, spec in graph_specs:
        precision = _precision_for_graph(requested_precision, kind)
        calibration_graph = calibration["graphs"][name]
        graphs[name] = {
            "kind": kind,
            "onnx": spec["onnx"],
            "onnx_sha256": calibration_graph["onnx_sha256"],
            "input_names": spec["input_names"],
            "precision": str(precision),
            "quantize_options": _quantize_options(precision, litemp_percentage),
            "calibration_samples": [
                {
                    "source_id": sample["source_id"],
                    "frame": sample["frame"],
                }
                for sample in calibration_graph["samples"]
            ],
            "status": "planned",
        }
    return {
        "format": RESULT_FORMAT,
        "graph_manifest_sha256": _sha256(graph_manifest_path),
        "calibration_manifest_sha256": _sha256(calibration_manifest_path),
        "requested_precision": requested_precision,
        "litemp_percentage": litemp_percentage,
        "graphs": graphs,
    }


def _check_resume(existing: dict[str, Any], planned: dict[str, Any]) -> None:
    for key in (
        "format",
        "graph_manifest_sha256",
        "calibration_manifest_sha256",
        "requested_precision",
        "litemp_percentage",
    ):
        if existing.get(key) != planned.get(key):
            raise ValueError(
                f"Cannot resume: {key} changed ({existing.get(key)!r} != {planned.get(key)!r})"
            )
    if set(existing["graphs"]) != set(planned["graphs"]):
        raise ValueError("Cannot resume: graph set changed")
    for name, graph in planned["graphs"].items():
        for key in ("onnx_sha256", "precision", "quantize_options"):
            if existing["graphs"][name].get(key) != graph[key]:
                raise ValueError(f"Cannot resume: {name} field {key} changed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-dir", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--precision",
        default="auto",
        choices=(
            "auto",
            "w8a16",
            "w8a16_mixed_fp16",
            "w8a16_mixed_int16",
        ),
        help=(
            "auto uses W8A16 for frontend/head and W8A16 mixed FP16 for "
            "Temporal/DepFormer (default: auto)"
        ),
    )
    parser.add_argument("--litemp-percentage", type=float, default=20.0)
    parser.add_argument(
        "--target-device",
        help="Exact AI Hub device name. If omitted, stop after quantization.",
    )
    parser.add_argument("--target-os", help="Optional exact device OS selector.")
    parser.add_argument("--source-model-manifest", type=Path)
    parser.add_argument("--minimum-sources", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--refresh-preflight",
        action="store_true",
        help="Ignore the cached receipt and fully re-hash/decompress every input.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and write the complete job plan without using AI Hub.",
    )
    args = parser.parse_args()
    if not 0 < args.litemp_percentage <= 100:
        raise SystemExit("--litemp-percentage must be within (0, 100]")
    if args.minimum_sources < 1:
        raise SystemExit("--minimum-sources must be positive")
    if args.target_os and not args.target_device:
        raise SystemExit("--target-os requires --target-device")

    graph_manifest_path = args.onnx_dir / "manifest.json"
    calibration_manifest_path = args.calibration_dir / "manifest.json"
    graph_manifest = json.loads(graph_manifest_path.read_text())
    calibration = json.loads(calibration_manifest_path.read_text())
    if graph_manifest.get("format") != GRAPH_SET_FORMAT:
        raise SystemExit(f"Unsupported graph set: {graph_manifest.get('format')}")
    if calibration.get("format") != CALIBRATION_FORMAT:
        raise SystemExit(f"Unsupported calibration set: {calibration.get('format')}")
    if calibration.get("graph_manifest_sha256") != _sha256(graph_manifest_path):
        raise SystemExit("Calibration does not match the ONNX graph manifest")

    graph_specs = _graph_specs(graph_manifest)
    planned = _build_result(
        graph_manifest_path,
        calibration_manifest_path,
        args.precision,
        args.litemp_percentage,
        graph_specs,
        calibration,
    )
    result_path = args.output_dir / "quantization_manifest.json"
    result_exists = result_path.exists()
    if result_exists:
        if not args.resume:
            raise SystemExit(f"{result_path} exists; pass --resume to continue it")
        result = json.loads(result_path.read_text())
        _check_resume(result, planned)
    else:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise SystemExit("Use an empty --output-dir or pass --resume")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        result = planned

    preflight_start = time.monotonic()
    preflight_cached = False
    if args.resume and not args.refresh_preflight:
        receipt = result.get("preflight_receipt")
        receipt_matches, receipt_reason = _preflight_receipt_matches(
            receipt,
            args.onnx_dir,
            args.calibration_dir,
            graph_specs,
            calibration,
            planned,
            args.minimum_sources,
        )
        if receipt_matches:
            preflight_cached = True
            print(
                f"[preflight] cache HIT: {receipt_reason}",
                flush=True,
            )
        elif receipt is None and result_exists:
            print(
                "[preflight] bootstrapping a receipt from the existing result; "
                "use --refresh-preflight for a forced full rescan",
                flush=True,
            )
            _validate_calibration(
                args.calibration_dir,
                calibration,
                graph_specs,
                args.minimum_sources,
                inspect_arrays=False,
            )
            result["preflight_receipt"] = _build_preflight_receipt(
                args.onnx_dir,
                args.calibration_dir,
                graph_specs,
                calibration,
                planned,
                args.minimum_sources,
                "legacy_result_bootstrap",
            )
            preflight_cached = True
        else:
            print(f"[preflight] cache MISS: {receipt_reason}", flush=True)

    if not preflight_cached:
        print(
            "[preflight] full validation: decompressing calibration arrays",
            flush=True,
        )
        _validate_calibration(
            args.calibration_dir,
            calibration,
            graph_specs,
            args.minimum_sources,
        )
        for graph_index, (name, _, spec) in enumerate(graph_specs, start=1):
            onnx_path = args.onnx_dir / spec["onnx"]
            print(
                f"[preflight] ONNX SHA256 {graph_index}/{len(graph_specs)}: {name}",
                flush=True,
            )
            actual_sha256 = _sha256(onnx_path)
            expected_sha256 = calibration["graphs"][name]["onnx_sha256"]
            if actual_sha256 != expected_sha256:
                raise SystemExit(f"ONNX SHA256 changed for {name}: {onnx_path}")
        result["preflight_receipt"] = _build_preflight_receipt(
            args.onnx_dir,
            args.calibration_dir,
            graph_specs,
            calibration,
            planned,
            args.minimum_sources,
            "full",
        )

    _write_json(result_path, result)
    print(
        f"[preflight] ready in {time.monotonic() - preflight_start:.2f}s",
        flush=True,
    )

    requested_target = None
    if args.target_device:
        requested_target = {"name": args.target_device}
        if args.target_os:
            requested_target["os"] = args.target_os
        existing_target = result.get("target_device")
        if existing_target is not None and existing_target != requested_target:
            raise SystemExit(
                f"Cannot resume with a different target device: "
                f"{existing_target} != {requested_target}"
            )
        existing_runtime = result.get("compile_runtime")
        if existing_runtime is not None and existing_runtime != COMPILE_RUNTIME.value:
            raise SystemExit(
                "Cannot resume with a different compile runtime: "
                f"{existing_runtime} != {COMPILE_RUNTIME.value}"
            )
        result["target_device"] = requested_target
        result["compile_runtime"] = COMPILE_RUNTIME.value
        _write_json(result_path, result)

    if args.dry_run:
        print(
            f"Dry-run PASS: {len(graph_specs)} graphs, "
            f"{len(calibration['sources'])} sources; plan={result_path}",
            flush=True,
        )
        return

    import qai_hub as hub

    reusable = _reusable_source_models(args.source_model_manifest)
    if args.source_model_manifest is not None and not reusable:
        print(
            "Source-model manifest has no model ID paired with an ONNX SHA256; "
            "its IDs will not be reused.",
            flush=True,
        )
    active_quantize_jobs: dict[str, Any] = {}
    failures: list[str] = []
    for name, _, spec in graph_specs:
        record = result["graphs"][name]
        target_state, recorded_job = _recheck_recorded_target(
            hub,
            record,
            stage="quantize",
            job_id_key="quantize_job_id",
            model_id_key="quantized_model_id",
        )
        if target_state == "success":
            print(f"quantize skip verified success: {name}", flush=True)
            _write_json(result_path, result)
            continue
        if target_state == "running":
            assert recorded_job is not None
            active_quantize_jobs[name] = recorded_job
            print(
                f"quantize resume running: {name} job={recorded_job.job_id}",
                flush=True,
            )
            _write_json(result_path, result)
            continue
        if record.get("status") in {
            "quantize_submit_failed",
            "quantize_failed",
            "quantize_record_invalid",
        }:
            record.pop("quantize_job_id", None)
            record.pop("error", None)
        try:
            if record.get("quantize_job_id"):
                job = hub.get_job(record["quantize_job_id"])
                print(f"quantize resume: {name} job={job.job_id}", flush=True)
            else:
                source_model_id = record.get("source_model_id") or reusable.get(
                    record["onnx_sha256"]
                )
                if source_model_id:
                    model = hub.get_model(source_model_id)
                    record["source_model_id"] = source_model_id
                    record["source_reused_by_sha256"] = True
                else:
                    model = hub.upload_model(str(args.onnx_dir / spec["onnx"]))
                    record["source_model_id"] = model.model_id
                    record["source_reused_by_sha256"] = False
                record["status"] = "source_ready"
                _write_json(result_path, result)
                entries = _load_calibration_entries(
                    args.calibration_dir, calibration["graphs"][name]
                )
                precision = _precision_for_graph(args.precision, record["kind"])
                job = hub.submit_quantize_job(
                    model=model,
                    calibration_data=entries,
                    activations_dtype=precision.activations_type,
                    weights_dtype=precision.weights_type,
                    name=f"moshi-{name}-{record['precision']}",
                    options=record["quantize_options"],
                )
                record["quantize_job_id"] = job.job_id
                record["status"] = "quantize_submitted"
                _write_json(result_path, result)
                del entries
                gc.collect()
                print(f"quantize submitted: {name} job={job.job_id}", flush=True)
            active_quantize_jobs[name] = job
        except Exception as error:
            record["status"] = "quantize_submit_failed"
            record["error"] = f"{type(error).__name__}: {error}"
            failures.append(name)
            _write_json(result_path, result)
            print(f"quantize submit FAILED: {name}: {error}", flush=True)

    for name, job in active_quantize_jobs.items():
        record = result["graphs"][name]
        try:
            job.wait()
            target = _required_target_model(job, "Quantize")
            record["quantized_model_id"] = target.model_id
            record["status"] = "quantize_succeeded"
            record.pop("error", None)
            print(f"quantize succeeded: {name} model={target.model_id}", flush=True)
        except Exception as error:
            record["status"] = "quantize_failed"
            record["error"] = f"{type(error).__name__}: {error}"
            failures.append(name)
            print(f"quantize FAILED: {name}: {error}", flush=True)
        _write_json(result_path, result)

    active_compile_jobs: dict[str, Any] = {}
    if args.target_device:
        assert requested_target is not None
        device_args = requested_target
        device = hub.Device(**device_args)
        for name, _, _ in graph_specs:
            record = result["graphs"][name]
            if not record.get("quantized_model_id"):
                continue
            target_state, recorded_job = _recheck_recorded_target(
                hub,
                record,
                stage="compile",
                job_id_key="compile_job_id",
                model_id_key="compiled_model_id",
            )
            if target_state == "success":
                print(f"compile skip verified success: {name}", flush=True)
                _write_json(result_path, result)
                continue
            if target_state == "running":
                assert recorded_job is not None
                active_compile_jobs[name] = recorded_job
                print(
                    f"compile resume running: {name} job={recorded_job.job_id}",
                    flush=True,
                )
                _write_json(result_path, result)
                continue
            if record.get("status") in {
                "compile_submit_failed",
                "compile_failed",
                "compile_record_invalid",
            }:
                record.pop("compile_job_id", None)
                record.pop("error", None)
            try:
                if record.get("compile_job_id"):
                    job = hub.get_job(record["compile_job_id"])
                    print(f"compile resume: {name} job={job.job_id}", flush=True)
                else:
                    job = hub.submit_compile_job(
                        model=hub.get_model(record["quantized_model_id"]),
                        input_specs=_input_specs(
                            args.calibration_dir, calibration["graphs"][name]
                        ),
                        device=device,
                        name=(
                            f"moshi-{name}-{record['precision']}-"
                            f"{COMPILE_RUNTIME.value.replace('_', '-')}"
                        ),
                        options=_compile_options(),
                    )
                    record["compile_runtime"] = COMPILE_RUNTIME.value
                    record["compile_job_id"] = job.job_id
                    record["status"] = "compile_submitted"
                    _write_json(result_path, result)
                    print(f"compile submitted: {name} job={job.job_id}", flush=True)
                active_compile_jobs[name] = job
            except Exception as error:
                record["status"] = "compile_submit_failed"
                record["error"] = f"{type(error).__name__}: {error}"
                failures.append(name)
                _write_json(result_path, result)
                print(f"compile submit FAILED: {name}: {error}", flush=True)

        for name, job in active_compile_jobs.items():
            record = result["graphs"][name]
            try:
                job.wait()
                target = _required_target_model(job, "Compile")
                record["compiled_model_id"] = target.model_id
                record["status"] = "compile_succeeded"
                record.pop("error", None)
                print(f"compile succeeded: {name} model={target.model_id}", flush=True)
            except Exception as error:
                record["status"] = "compile_failed"
                record["error"] = f"{type(error).__name__}: {error}"
                failures.append(name)
                print(f"compile FAILED: {name}: {error}", flush=True)
            _write_json(result_path, result)

    if failures:
        unique_failures = sorted(set(failures))
        raise SystemExit(
            f"Batch finished with {len(unique_failures)} failed graphs: {unique_failures}; "
            f"resume with --resume after addressing the errors"
        )
    stage = "compile" if args.target_device else "quantize"
    print(f"Moshi LM graph-set {stage}: PASS manifest={result_path}", flush=True)
    print(
        "This proves AI Hub job completion only; run chained clean/overlap "
        "inference before making numerical-accuracy or deployment claims.",
        flush=True,
    )


if __name__ == "__main__":
    main()

"""Compile and numerically probe two unquantized, stateful Mimi ONNX graphs.

Run after moshi_export_mimi_streaming_onnx.py passes. Jobs and outputs are
recorded separately for each component and frame so interrupted runs resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import qai_hub as hub

from moshi_export_mimi_streaming_onnx import (
    FRAME_SAMPLES,
    FORMAT,
    INPUT_NAMES,
    STATE_OUTPUT_NAMES,
    _load_audio,
    _session,
    _write_wav,
)

OPTIONS = "--target_runtime qnn_dlc --truncate_64bit_io"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save(path: Path, record: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(path)


def _metrics(actual: np.ndarray, expected: np.ndarray) -> dict:
    actual, expected = np.asarray(actual), np.asarray(expected)
    result = {"shape": list(actual.shape), "expected_shape": list(expected.shape),
              "finite": bool(np.isfinite(actual).all())}
    if actual.shape != expected.shape or not result["finite"]:
        result["pass"] = False
        return result
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    result["max_abs"] = float(np.abs(delta).max())
    result["rmse"] = float(np.sqrt(np.mean(delta ** 2)))
    if np.issubdtype(expected.dtype, np.integer):
        result["matching"] = int(np.count_nonzero(actual == expected))
        result["total"] = int(expected.size)
        result["pass"] = bool(np.array_equal(actual, expected))
    else:
        reference_rms = float(np.sqrt(np.mean(expected.astype(np.float64) ** 2)))
        result["relative_rms"] = result["rmse"] / max(reference_rms, 1e-12)
        result["pass"] = bool(result["rmse"] <= 0.01 and result["max_abs"] <= 0.1)
    return result


def _qnn_input(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    if value.dtype == np.int64:
        if value.size and (value.min() < np.iinfo(np.int32).min or
                           value.max() > np.iinfo(np.int32).max):
            raise ValueError("QNN int32 position overflow")
        return value.astype(np.int32)
    return value


def _reference(session, frames: list[np.ndarray], data_name: str, state: list[np.ndarray]):
    names = [data_name, *INPUT_NAMES]
    outputs = ["codes" if data_name == "audio" else "audio", *STATE_OUTPUT_NAMES]
    current = state
    results = []
    for data in frames:
        values = session.run(outputs, dict(zip(names, [data, *current], strict=True)))
        results.append(values)
        current = values[1:]
    return results


def _compile(component: str, graph: Path, device: dict, directory: Path):
    source = onnx.load(str(graph), load_external_data=False)
    graph_inputs = {item.name: (tuple(dim.dim_value for dim in item.type.tensor_type.shape.dim),
                                onnx.helper.tensor_dtype_to_np_dtype(item.type.tensor_type.elem_type))
                    for item in source.graph.input}
    names = ["audio" if component == "encoder" else "codes", *INPUT_NAMES]
    if set(graph_inputs) != set(names):
        raise ValueError(f"Unexpected {component} graph inputs: {list(graph_inputs)}")
    record_path = directory / f"cloud_{component}.json"
    config = {"onnx_sha256": _sha256(graph), "device": device, "options": OPTIONS}
    record = json.loads(record_path.read_text()) if record_path.exists() else {"config": config}
    if record["config"] != config:
        raise ValueError("Cloud config changed; choose a new output directory")
    if "source_model_id" not in record:
        record["source_model_id"] = hub.upload_model(str(graph)).model_id
        _save(record_path, record)
    if "compile_job_id" not in record:
        specs = {name: (graph_inputs[name][0], str(np.dtype(graph_inputs[name][1])))
                 for name in names}
        job = hub.submit_compile_job(
            model=hub.get_model(record["source_model_id"]),
            input_specs=specs, device=hub.Device(**device), options=OPTIONS,
            name=f"moshi-mimi-streaming-{component}-float",
        )
        record["compile_job_id"] = job.job_id
        _save(record_path, record)
    job = hub.get_job(record["compile_job_id"])
    print(f"{component} compile={job.url}", flush=True)
    job.wait()
    if not job.get_status().success:
        raise RuntimeError(f"{component} compilation failed: {job.url}")
    target = job.get_target_model()
    if target is None:
        raise RuntimeError(f"{component} compile returned no model")
    if set(job.get_target_shapes()) != set(names):
        raise RuntimeError(f"{component} compiled input names differ: {job.get_target_shapes()}")
    record["compiled_model_id"] = target.model_id
    _save(record_path, record)
    return target, record, record_path, names


def _infer(component, target, record, record_path, names, frame, data, state, expected,
           device, directory, diagnostic_continue):
    archive = directory / f"{component}_qnn_frame_{frame}.npz"
    entry = record.setdefault("inference", {}).setdefault(f"frame_{frame}", {})
    inputs = dict(zip(names, [data, *state], strict=True))
    if archive.exists():
        with np.load(archive) as saved:
            for name, value in inputs.items():
                if not np.array_equal(saved[f"input_{name}"], _qnn_input(value)):
                    raise ValueError(f"Cached {component} frame {frame} input {name} changed")
            actual = [saved[f"output_{index}"].copy() for index in range(4)]
    else:
        if "job_id" not in entry:
            job = hub.submit_inference_job(
                model=target, device=hub.Device(**device),
                inputs={name: [_qnn_input(value)] for name, value in inputs.items()},
                name=f"moshi-mimi-streaming-{component}-frame-{frame}",
            )
            entry["job_id"] = job.job_id
            _save(record_path, record)
        job = hub.get_job(entry["job_id"])
        print(f"{component} frame={frame} inference={job.url}", flush=True)
        job.wait()
        if not job.get_status().success:
            raise RuntimeError(f"{component} inference failed: {job.url}")
        downloaded = job.download_output_data()
        expected_names = ["codes" if component == "encoder" else "audio", *STATE_OUTPUT_NAMES]
        if set(downloaded) == set(expected_names):
            keys = expected_names
        elif set(downloaded) == {f"output_{index}" for index in range(4)}:
            keys = [f"output_{index}" for index in range(4)]
        else:
            raise ValueError(f"Unexpected {component} outputs: {list(downloaded)}")
        if any(len(downloaded[key]) != 1 for key in keys):
            raise ValueError("Expected one cloud output sample")
        actual = [np.asarray(downloaded[key][0]) for key in keys]
        np.savez_compressed(archive, **{f"input_{name}": _qnn_input(value)
                                     for name, value in inputs.items()},
                            **{f"output_{index}": value for index, value in enumerate(actual)})
    comparisons = {name: _metrics(value, reference) for name, value, reference in
                   zip(["data", *STATE_OUTPUT_NAMES], actual, expected, strict=True)}
    entry["comparison"] = comparisons
    _save(record_path, record)
    print(f"{component} frame={frame}: {comparisons}", flush=True)
    if component == "decoder":
        _write_wav(directory / f"decoder_qnn_frame_{frame}.wav", actual[0])
    if not all(item["pass"] for item in comparisons.values()) and not diagnostic_continue:
        raise RuntimeError(f"{component} frame {frame} numeric parity failed")
    return actual[1:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-dir", type=Path, required=True)
    parser.add_argument("--audio-wav", type=Path, required=True)
    parser.add_argument("--device-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--component", choices=("encoder", "decoder"), required=True)
    parser.add_argument("--diagnostic-continue", action="store_true",
                        help="Chain QNN states despite a numeric failure; overall result still fails")
    args = parser.parse_args()
    manifest = json.loads((args.onnx_dir / "manifest.json").read_text())
    if manifest.get("format") != FORMAT or manifest.get("frames_verified") != 2:
        raise ValueError("A passing two-frame explicit-state ONNX export is required")
    device = json.loads(args.device_manifest.read_text())["target_device"]
    if device != {"name": "Samsung Galaxy S26 (Family)", "os": "16"}:
        raise ValueError(f"Unexpected device from existing manifest: {device}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audio = _load_audio(args.audio_wav).numpy()
    audio_frames = [audio[..., i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES] for i in range(2)]
    encoder_graph = args.onnx_dir / manifest["encoder"]["onnx"]
    decoder_graph = args.onnx_dir / manifest["decoder"]["onnx"]
    initial_encoder = [np.zeros(1, np.int64),
                       np.zeros(manifest["encoder"]["state_spec"]["kv_cache_shape"], np.float32),
                       np.zeros((1, sum(np.prod(shape) for shape in
                                         manifest["encoder"]["state_spec"]["conv_state_shapes"])), np.float32)]
    encoder_reference = _reference(_session(encoder_graph), audio_frames, "audio", initial_encoder)
    frames = audio_frames if args.component == "encoder" else [item[0] for item in encoder_reference]
    initial = initial_encoder if args.component == "encoder" else [
        np.zeros(1, np.int64),
        np.zeros(manifest["decoder"]["state_spec"]["kv_cache_shape"], np.float32),
        np.zeros((1, sum(np.prod(shape) for shape in
                          manifest["decoder"]["state_spec"]["conv_state_shapes"])), np.float32)]
    reference = encoder_reference if args.component == "encoder" else _reference(
        _session(decoder_graph), frames, "codes", initial)
    graph = encoder_graph if args.component == "encoder" else decoder_graph
    target, record, record_path, names = _compile(args.component, graph, device, args.output_dir)
    current = initial
    for frame, (data, expected) in enumerate(zip(frames, reference, strict=True)):
        current = _infer(args.component, target, record, record_path, names,
                         frame, data, current, expected, device, args.output_dir,
                         args.diagnostic_continue)
    if any(not item["pass"] for entry in record["inference"].values()
           for item in entry["comparison"].values()):
        raise RuntimeError(f"Mimi streaming {args.component} two-frame float QNN parity: FAIL")
    print(f"Mimi streaming {args.component} two-frame float QNN numeric parity: PASS", flush=True)


if __name__ == "__main__":
    main()

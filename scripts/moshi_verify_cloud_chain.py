"""Compare multi-frame full Temporal ONNX and cloud chains with resumable jobs."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import qai_hub as hub
import torch


def save_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def metrics(actual: np.ndarray, expected: np.ndarray) -> dict:
    if actual.shape != expected.shape:
        raise ValueError(f"Shape mismatch: {actual.shape} != {expected.shape}")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Non-finite result")
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    rmse = float(np.sqrt(np.mean(delta ** 2)))
    reference_rms = float(np.sqrt(np.mean(expected.astype(np.float64) ** 2)))
    return {"max_abs": float(np.abs(delta).max()), "rmse": rmse,
            "relative_rms": rmse / max(reference_rms, 1e-12)}


def cloud_step(target_id: str, inputs: dict, output_names: list[str], stem: Path) -> list:
    digest = hashlib.sha256(target_id.encode())
    digest.update(json.dumps(output_names).encode())
    for name, value in inputs.items():
        digest.update(f"{name}:{value.shape}:{value.dtype}".encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    signature = digest.hexdigest()
    record_path = stem.with_suffix(".json")
    archive_path = stem.with_suffix(".npz")
    if record_path.exists():
        record = json.loads(record_path.read_text())
        if record["signature"] != signature:
            raise ValueError(f"Inputs/model changed; use a new run directory: {stem}")
        job = hub.get_job(record["job_id"])
    else:
        job = hub.submit_inference_job(
            model=hub.get_model(target_id),
            device=hub.Device(name="Samsung Galaxy S26 (Family)", os="16"),
            inputs={name: [value] for name, value in inputs.items()},
            name=f"moshi-chain-{stem.name}",
        )
        save_json(record_path, {"signature": signature, "job_id": job.job_id,
                                "target_model_id": target_id})
    print(f"{stem.name}: job={job.job_id}", flush=True)
    if archive_path.exists():
        with np.load(archive_path) as archive:
            return [archive[name].copy() for name in output_names]
    job.wait()
    downloaded = job.download_output_data()
    if downloaded is None:
        raise RuntimeError(f"No outputs for {job.job_id}; inspect the cloud job")
    named = set(output_names)
    numbered = {f"output_{index}" for index in range(len(output_names))}
    if set(downloaded) == named:
        keys = output_names
    elif set(downloaded) == numbered:
        keys = [f"output_{index}" for index in range(len(output_names))]
    else:
        raise ValueError(f"Unexpected outputs: {list(downloaded)}")
    if any(len(downloaded[key]) != 1 for key in keys):
        raise ValueError("Expected one sample per output")
    outputs = [np.asarray(downloaded[key][0]) for key in keys]
    temporary = archive_path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **dict(zip(output_names, outputs, strict=True)))
    temporary.replace(archive_path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--input-hidden", type=Path, help="NPY array [frames, 1, 1, hidden_dim] of real replay embeddings")
    parser.add_argument("--target-override", help="Target model ID for the first 0:2 shard")
    parser.add_argument("--quantized", action="store_true", help="Allow bounded numerical changes in untouched cache regions")
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")
    root = args.export_dir
    manifest = json.loads((root / "manifest.json").read_text())
    shards = sorted(manifest["shards"], key=lambda item: item["start_layer"])
    records = json.loads((root / "compiled_shards.json").read_text())
    targets = {(item["start"], item["end"]): item["target_model_id"] for item in records}
    cursor = 0
    for shard in shards:
        start, end = shard["start_layer"], shard["end_layer_exclusive"]
        if start != cursor or end <= start or (start, end) not in targets:
            raise ValueError("Expected contiguous compiled shards from layer zero")
        cursor = end
    if cursor != manifest["total_temporal_layers"] or cursor != 32:
        raise ValueError("Expected all 32 Temporal layers")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    references = torch.load(root / shards[0]["reference"], map_location="cpu", weights_only=True)
    if args.input_hidden is None:
        if len(references) < args.frames:
            raise ValueError("Not enough saved frames; provide --input-hidden from moshi_prepare_chain_inputs.py")
        cpu_hidden = [references[frame]["inputs"]["hidden"].numpy().copy() for frame in range(args.frames)]
    else:
        inputs = np.load(args.input_hidden, allow_pickle=False)
        expected_shape = tuple(references[0]["inputs"]["hidden"].shape)
        if inputs.ndim != 4 or inputs.shape[0] < args.frames or inputs.shape[1:] != expected_shape:
            raise ValueError(f"Expected at least {args.frames} embeddings with shape {expected_shape}")
        if inputs.dtype != np.float32 or not np.isfinite(inputs[:args.frames]).all():
            raise ValueError("Embeddings must be finite FP32")
        cpu_hidden = [value.copy() for value in inputs[:args.frames]]
    cloud_hidden = [hidden.copy() for hidden in cpu_hidden]
    del references
    reports = []
    for shard in shards:
        start, end = shard["start_layer"], shard["end_layer_exclusive"]
        references = torch.load(root / shard["reference"], map_location="cpu", weights_only=True)
        cache_names = shard["input_names"][2:]
        if shard["input_names"][:2] != ["hidden", "position"]:
            raise ValueError("Unexpected shard input order")
        cpu_cache = [references[0]["inputs"][name].numpy().copy() for name in cache_names]
        if any(np.count_nonzero(value) for value in cpu_cache):
            raise ValueError("Expected initially empty caches")
        cloud_cache = [value.copy() for value in cpu_cache]
        del references
        session = ort.InferenceSession(str(root / shard["onnx"]), providers=["CPUExecutionProvider"])
        for frame in range(args.frames):
            position = np.array([frame], dtype=np.int32)
            cpu_inputs = dict(zip(shard["input_names"],
                                  [cpu_hidden[frame], position, *cpu_cache], strict=True))
            expected = session.run(shard["output_names"], cpu_inputs)
            cloud_inputs = dict(zip(shard["input_names"],
                                    [cloud_hidden[frame], position, *cloud_cache], strict=True))
            stem = args.run_dir / f"layers_{start}_{end}_frame_{frame}"
            target_id = args.target_override if args.target_override and (start, end) == (0, 2) else targets[(start, end)]
            actual = cloud_step(target_id, cloud_inputs, shard["output_names"], stem)
            result = {"start": start, "end": end, "frame": frame,
                      "hidden": metrics(actual[0], expected[0]), "cache": {}}
            for name, received, reference, previous in zip(
                cache_names, actual[1:], expected[1:], cloud_cache, strict=True
            ):
                if received.shape != reference.shape or not np.isfinite(received).all():
                    raise ValueError(f"Invalid cache: {name}")
                slot = frame % received.shape[2]
                for lower, upper in ((0, slot), (slot + 1, received.shape[2])):
                    if lower >= upper:
                        continue
                    untouched_actual = received[:, :, lower:upper]
                    untouched_previous = previous[:, :, lower:upper]
                    if args.quantized:
                        untouched_error = metrics(untouched_actual, untouched_previous)
                        if untouched_error["max_abs"] > 0.25:
                            raise ValueError(f"Untouched quantized cache changed too much: {stem.name} {name}: {untouched_error}")
                    elif not np.array_equal(untouched_actual, untouched_previous):
                        raise ValueError(f"Untouched cache changed: {stem.name} {name}")
                result["cache"][name] = metrics(received[:, :, slot], reference[:, :, slot])
            print(json.dumps(result), flush=True)
            reports.append(result)
            save_json(args.run_dir / "metrics.json", reports)
            cpu_hidden[frame], cpu_cache = expected[0], expected[1:]
            cloud_hidden[frame], cloud_cache = actual[0], actual[1:]
        del session, cpu_cache, cloud_cache, cpu_inputs, cloud_inputs, actual, expected
        gc.collect()
    np.savez(args.run_dir / "final_hidden.npz",
             cpu=np.stack(cpu_hidden), cloud=np.stack(cloud_hidden))
    print(f"Completed 32-layer, {args.frames}-frame chains; untouched cloud caches: PASS.")
    print("Numerical errors are in metrics.json; no automatic accuracy acceptance threshold applied.")
    print("Embedding inputs are saved replay inputs; final norm/text head, DepFormer and Mimi are not included.")


if __name__ == "__main__":
    main()

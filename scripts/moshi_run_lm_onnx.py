"""Run a complete exported Moshi LM graph set with ONNX Runtime only."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch


def _session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def _metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    if actual.shape != expected.shape:
        raise ValueError(f"Shape mismatch: {actual.shape} != {expected.shape}")
    if not np.isfinite(actual).all():
        raise ValueError("ONNX output contains NaN or Inf")
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-dir", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int)
    args = parser.parse_args()
    manifest_path = args.onnx_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Missing {manifest_path}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit("Use an empty --output-dir")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != "moshi-lm-onnx-graph-set-v1":
        raise SystemExit(f"Unsupported manifest format: {manifest.get('format')}")

    sequence = torch.load(
        args.trace_dir / "temporal_sequence.pt",
        map_location="cpu",
        weights_only=True,
    ).numpy()
    frames = args.frames or sequence.shape[-1]
    if frames < 2 or frames > sequence.shape[-1]:
        raise SystemExit(f"--frames must be within 2:{sequence.shape[-1]}")
    sequence = sequence[..., :frames]

    frontend = manifest["frontend"]
    session = _session(args.onnx_dir / frontend["onnx"])
    hidden_frames = [
        session.run(frontend["output_names"], {"sequence": sequence[..., frame : frame + 1]})[0]
        for frame in range(frames)
    ]
    del session
    gc.collect()
    print(f"frontend: PASS frames={frames}", flush=True)

    for shard in manifest["temporal_shards"]:
        session = _session(args.onnx_dir / shard["onnx"])
        caches = [np.zeros(shape, np.float32) for shape in shard["cache_shapes"]]
        outputs = []
        for frame, hidden in enumerate(hidden_frames):
            position = np.array([frame], dtype=np.int64)
            values = [hidden, position, *caches]
            received = session.run(
                shard["output_names"],
                dict(zip(shard["input_names"], values, strict=True)),
            )
            if not all(np.isfinite(value).all() for value in received):
                raise RuntimeError(
                    f"Non-finite output in layers "
                    f"{shard['start_layer']}:{shard['end_layer_exclusive']} frame={frame}"
                )
            outputs.append(received[0])
            caches = received[1:]
        hidden_frames = outputs
        print(
            f"layers={shard['start_layer']}:{shard['end_layer_exclusive']} "
            f"frames={frames}: PASS",
            flush=True,
        )
        del session, caches
        gc.collect()

    head = manifest["head"]
    session = _session(args.onnx_dir / head["onnx"])
    temporal_frames = []
    text_logits_frames = []
    for hidden in hidden_frames:
        temporal, text_logits = session.run(head["output_names"], {"hidden": hidden})
        temporal_frames.append(temporal)
        text_logits_frames.append(text_logits)
    del session
    gc.collect()
    text_logits = np.concatenate(text_logits_frames, axis=2)
    text_tokens = np.argmax(text_logits, axis=-1).astype(np.int64)
    print(f"text head: PASS shape={text_logits.shape}", flush=True)

    previous_tokens = [
        np.argmax(value, axis=-1)[:, 0, 0].astype(np.int64)
        for value in text_logits_frames
    ]
    cache_frames: list[list[np.ndarray]] | None = None
    audio_tokens_by_frame: list[list[np.ndarray]] = [[] for _ in range(frames)]
    audio_logits_by_frame: list[list[np.ndarray]] = [[] for _ in range(frames)]
    for step in manifest["depformer"]["steps"]:
        session = _session(args.onnx_dir / step["onnx"])
        if cache_frames is None:
            cache_frames = [
                [np.zeros(shape, np.float32) for shape in step["cache_shapes"]]
                for _ in range(frames)
            ]
        next_tokens = []
        next_caches = []
        for frame in range(frames):
            values = [previous_tokens[frame], temporal_frames[frame], *cache_frames[frame]]
            received = session.run(
                step["output_names"],
                dict(zip(step["input_names"], values, strict=True)),
            )
            if not all(np.isfinite(value).all() for value in received):
                raise RuntimeError(
                    f"Non-finite DepFormer output codebook={step['codebook']} frame={frame}"
                )
            next_tokens.append(received[0])
            next_caches.append(received[2:])
            audio_tokens_by_frame[frame].append(received[0][:, None])
            audio_logits_by_frame[frame].append(received[1])
        previous_tokens = next_tokens
        cache_frames = next_caches
        print(f"DepFormer codebook={step['codebook']}: PASS", flush=True)
        del session
        gc.collect()

    frame_tokens = [np.concatenate(values, axis=1) for values in audio_tokens_by_frame]
    audio_tokens = np.stack(frame_tokens, axis=2)
    frame_logits = [np.concatenate(values, axis=1) for values in audio_logits_by_frame]
    audio_logits = np.stack(frame_logits, axis=1)
    temporal = np.concatenate(temporal_frames, axis=1)

    reference_text_logits = _load_trace_tensor(args.trace_dir, "text_logits")[..., :frames, :]
    reference_audio_logits = _load_trace_tensor(args.trace_dir, "depformer_logits")[:, :frames]
    reference_text_tokens = np.argmax(reference_text_logits, axis=-1)
    reference_audio_tokens = np.argmax(reference_audio_logits, axis=-1)[..., 0]
    reference_audio_tokens = np.transpose(reference_audio_tokens, (0, 2, 1))
    report = {
        "frames": frames,
        "text_logits": _metrics(text_logits, reference_text_logits),
        "audio_logits": _metrics(audio_logits, reference_audio_logits),
        "text_token_agreement": float(np.mean(text_tokens == reference_text_tokens)),
        "audio_token_agreement": float(np.mean(audio_tokens == reference_audio_tokens)),
        "shapes": {
            "temporal": list(temporal.shape),
            "text_logits": list(text_logits.shape),
            "text_tokens": list(text_tokens.shape),
            "audio_logits": list(audio_logits.shape),
            "audio_tokens": list(audio_tokens.shape),
        },
    }
    args.output_dir.mkdir(parents=True)
    np.savez_compressed(
        args.output_dir / "onnx_outputs.npz",
        temporal=temporal,
        text_logits=text_logits,
        text_tokens=text_tokens,
        audio_logits=audio_logits,
        audio_tokens=audio_tokens,
    )
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if report["text_token_agreement"] != 1.0 or report["audio_token_agreement"] != 1.0:
        raise RuntimeError(f"ONNX token mismatch against BF16 reference: {report}")
    print("Standalone ONNX LM inference and BF16 token parity: PASS", flush=True)


if __name__ == "__main__":
    main()

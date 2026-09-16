"""Run a complete exported Moshi LM graph set with ONNX Runtime only."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch

CALIBRATION_FORMAT = "moshi-lm-graph-calibration-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    return specs


def _parse_frame_positions(value: str) -> list[int]:
    try:
        positions = sorted({int(part.strip()) for part in value.split(",")})
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "frame positions must be comma-separated integers"
        ) from error
    if not positions or positions[0] < 0:
        raise argparse.ArgumentTypeError("frame positions must be non-negative")
    return positions


class CalibrationCapture:
    """Persist real per-graph ORT feeds selected from sequential LM execution."""

    def __init__(
        self,
        output_dir: Path,
        onnx_dir: Path,
        graph_manifest: dict[str, Any],
        source_id: str,
        trace_dir: Path,
        frame_positions: list[int],
    ) -> None:
        self.output_dir = output_dir
        self.source_id = source_id
        self.frame_positions = set(frame_positions)
        self.manifest_path = output_dir / "manifest.json"
        self.specs = {
            name: (kind, spec)
            for name, kind, spec in _graph_specs(graph_manifest)
        }
        graph_manifest_path = onnx_dir / "manifest.json"
        graph_manifest_sha256 = _sha256(graph_manifest_path)
        if self.manifest_path.exists():
            self.manifest = json.loads(self.manifest_path.read_text())
            if self.manifest.get("format") != CALIBRATION_FORMAT:
                raise ValueError(
                    f"Unsupported calibration manifest: {self.manifest.get('format')}"
                )
            if self.manifest.get("graph_manifest_sha256") != graph_manifest_sha256:
                raise ValueError(
                    "Calibration directory belongs to a different ONNX graph manifest"
                )
        else:
            self.manifest = {
                "format": CALIBRATION_FORMAT,
                "graph_manifest_sha256": graph_manifest_sha256,
                "graphs": {},
                "sources": [],
            }

        graphs = self.manifest["graphs"]
        for name, (kind, spec) in self.specs.items():
            onnx_path = onnx_dir / spec["onnx"]
            expected = {
                "kind": kind,
                "onnx": spec["onnx"],
                "onnx_sha256": _sha256(onnx_path),
                "input_names": spec["input_names"],
            }
            if name in graphs:
                for key, value in expected.items():
                    if graphs[name].get(key) != value:
                        raise ValueError(
                            f"Calibration graph {name!r} changed at field {key!r}"
                        )
            else:
                graphs[name] = {**expected, "samples": []}

        source = {
            "id": source_id,
            "trace_dir": str(trace_dir.resolve()),
            "frame_positions": frame_positions,
        }
        previous = next(
            (item for item in self.manifest["sources"] if item["id"] == source_id),
            None,
        )
        if previous is not None and previous != source:
            raise ValueError(
                f"Calibration source ID {source_id!r} already describes another trace"
            )
        if previous is None:
            self.manifest["sources"].append(source)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._write_manifest()

    def _write_manifest(self) -> None:
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.manifest, indent=2) + "\n")
        temporary.replace(self.manifest_path)

    def capture(self, graph_name: str, frame: int, feed: dict[str, np.ndarray]) -> None:
        if frame not in self.frame_positions:
            return
        graph = self.manifest["graphs"][graph_name]
        if list(feed) != graph["input_names"]:
            raise ValueError(
                f"{graph_name} feed names {list(feed)} != {graph['input_names']}"
            )
        safe_source = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.source_id).strip("_")
        source_hash = hashlib.sha256(self.source_id.encode()).hexdigest()[:8]
        safe_source = f"{safe_source or 'source'}_{source_hash}"
        relative = Path("samples") / graph_name / f"{safe_source}_frame_{frame:04d}.npz"
        path = self.output_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp.npz")
        np.savez_compressed(temporary, **feed)
        temporary.replace(path)
        sample = {
            "source_id": self.source_id,
            "frame": frame,
            "file": str(relative),
        }
        samples = graph["samples"]
        previous_index = next(
            (
                index
                for index, item in enumerate(samples)
                if item["source_id"] == self.source_id and item["frame"] == frame
            ),
            None,
        )
        if previous_index is None:
            samples.append(sample)
            samples.sort(key=lambda item: (item["source_id"], item["frame"]))
        else:
            samples[previous_index] = sample
        self._write_manifest()


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
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        help="Append selected real ORT feeds to this calibration dataset.",
    )
    parser.add_argument(
        "--calibration-frames",
        type=_parse_frame_positions,
        default=_parse_frame_positions("0,1,2,4,8,16,24,37"),
        help="Comma-separated frame positions to capture (default: 0,1,2,4,8,16,24,37).",
    )
    parser.add_argument(
        "--calibration-source-id",
        help="Stable sample source label; defaults to the trace directory name.",
    )
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
    capture = None
    if args.calibration_dir is not None:
        invalid = [frame for frame in args.calibration_frames if frame >= frames]
        if invalid:
            raise SystemExit(
                f"Calibration frames {invalid} are unavailable in a {frames}-frame trace"
            )
        capture = CalibrationCapture(
            args.calibration_dir,
            args.onnx_dir,
            manifest,
            args.calibration_source_id or args.trace_dir.name,
            args.trace_dir,
            args.calibration_frames,
        )

    frontend = manifest["frontend"]
    session = _session(args.onnx_dir / frontend["onnx"])
    hidden_frames = []
    for frame in range(frames):
        feed = {"sequence": sequence[..., frame : frame + 1]}
        if capture is not None:
            capture.capture("frontend", frame, feed)
        hidden_frames.append(session.run(frontend["output_names"], feed)[0])
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
            feed = dict(zip(shard["input_names"], values, strict=True))
            if capture is not None:
                capture.capture(Path(shard["onnx"]).stem, frame, feed)
            received = session.run(
                shard["output_names"],
                feed,
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
    for frame, hidden in enumerate(hidden_frames):
        feed = {"hidden": hidden}
        if capture is not None:
            capture.capture("head", frame, feed)
        temporal, text_logits = session.run(head["output_names"], feed)
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
            feed = dict(zip(step["input_names"], values, strict=True))
            if capture is not None:
                capture.capture(Path(step["onnx"]).stem, frame, feed)
            received = session.run(
                step["output_names"],
                feed,
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
    if capture is not None:
        print(
            f"Calibration capture: PASS dir={args.calibration_dir} "
            f"source={capture.source_id} frames={sorted(capture.frame_positions)}",
            flush=True,
        )


if __name__ == "__main__":
    main()

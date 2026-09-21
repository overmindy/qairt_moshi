"""Export two fixed-shape Mimi frame graphs with explicit streaming state.

The graph shapes are static for QNN: batch one, one 80 ms frame, eight
codebooks, and preallocated 250-step Transformer caches. State values are
inputs and outputs and are chained between frames by the host.
"""

from __future__ import annotations

import argparse
import json
import sys
import wave
from array import array
from dataclasses import asdict
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qai_hub_models.models.templates.moshi.explicit_mimi import (  # noqa: E402
    ExplicitMimiDecoder,
    ExplicitMimiEncoder,
)
from qai_hub_models.models.templates.moshi.external_repos.moshi.moshi.moshi.utils.compile import (  # noqa: E501
    no_compile,
)
from qai_hub_models.models.templates.moshi.model import (  # noqa: E402
    FRAME_SAMPLES,
    resolve_moshi_checkpoint,
)

FORMAT = "moshi-mimi-explicit-streaming-onnx-v1"
INPUT_NAMES = ["position", "kv_cache", "conv_state"]
STATE_OUTPUT_NAMES = ["position_out", "kv_cache_out", "conv_state_out"]


def _load_audio(path: Path) -> torch.Tensor:
    with wave.open(str(path), "rb") as source:
        properties = (
            source.getnchannels(),
            source.getsampwidth(),
            source.getframerate(),
            source.getcomptype(),
        )
        expected = (1, 2, 24_000, "NONE")
        if properties != expected:
            raise ValueError(f"Expected {expected}, got {properties}")
        if source.getnframes() < 2 * FRAME_SAMPLES:
            raise ValueError("Two Mimi frames are required")
        samples = array("h")
        samples.frombytes(source.readframes(2 * FRAME_SAMPLES))
        if sys.byteorder != "little":
            samples.byteswap()
    return torch.tensor(samples, dtype=torch.float32).reshape(1, 1, -1) / 32768


def _write_wav(path: Path, audio: np.ndarray) -> None:
    pcm = np.rint(np.clip(audio.reshape(-1), -1, 32767 / 32768) * 32768).astype("<i2")
    with wave.open(str(path), "wb") as destination:
        destination.setnchannels(1)
        destination.setsampwidth(2)
        destination.setframerate(24_000)
        destination.writeframes(pcm.tobytes())


def _assert_equal(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise RuntimeError(
            f"{name}: shape/dtype {tuple(actual.shape)}/{actual.dtype} != "
            f"{tuple(expected.shape)}/{expected.dtype}"
        )
    maximum = float((actual.float() - expected.float()).abs().max())
    print(f"{name}: exact={torch.equal(actual, expected)} max_abs={maximum:.8g}", flush=True)
    if not torch.equal(actual, expected):
        raise RuntimeError(f"{name}: explicit state wrapper differs from upstream streaming")


def _export(
    module: torch.nn.Module,
    data: torch.Tensor,
    state: tuple[torch.Tensor, ...],
    path: Path,
    data_name: str,
    output_name: str,
) -> None:
    with no_compile():
        torch.onnx.export(
            module,
            (data, *state),
            str(path),
            opset_version=17,
            dynamo=False,
            do_constant_folding=True,
            input_names=[data_name, *INPUT_NAMES],
            output_names=[output_name, *STATE_OUTPUT_NAMES],
        )
    onnx.checker.check_model(str(path))
    print(f"ONNX checker PASS: {path}", flush=True)


def _session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def _ort_chain(
    path: Path,
    data_name: str,
    output_name: str,
    frames: list[np.ndarray],
    state: tuple[torch.Tensor, ...],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    session = _session(path)
    expected_inputs = [data_name, *INPUT_NAMES]
    expected_outputs = [output_name, *STATE_OUTPUT_NAMES]
    if [item.name for item in session.get_inputs()] != expected_inputs:
        actual = [item.name for item in session.get_inputs()]
        raise RuntimeError(f"Unexpected ONNX inputs: {actual}")
    if [item.name for item in session.get_outputs()] != expected_outputs:
        actual = [item.name for item in session.get_outputs()]
        raise RuntimeError(f"Unexpected ONNX outputs: {actual}")
    current = [value.detach().cpu().numpy() for value in state]
    outputs = []
    for frame, data in enumerate(frames):
        values = session.run(
            expected_outputs,
            dict(zip(expected_inputs, [data, *current], strict=True)),
        )
        outputs.append(values[0])
        current = values[1:]
        print(
            f"ORT {path.stem} frame={frame} position={int(current[0].item())}",
            flush=True,
        )
    return outputs, current


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--audio-wav", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit("Use a new empty Mimi explicit-state output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    audio = _load_audio(args.audio_wav).to(args.device)
    checkpoint = resolve_moshi_checkpoint(model_dir=args.model_dir)
    mimi = checkpoint.get_mimi(device=args.device)
    audio_frames = [
        audio[..., frame * FRAME_SAMPLES : (frame + 1) * FRAME_SAMPLES]
        for frame in range(2)
    ]

    upstream_codes = []
    upstream_audio = []
    with no_compile(), mimi.streaming(1):
        for frame in audio_frames:
            codes = mimi.encode(frame).to(torch.int32)
            upstream_codes.append(codes)
            upstream_audio.append(mimi.decode(codes)[..., :FRAME_SAMPLES])

    encoder = ExplicitMimiEncoder(mimi)
    encoder_initial = encoder.initial_state()
    encoder_state = tuple(value.clone() for value in encoder_initial)
    explicit_codes = []
    for frame, expected in zip(audio_frames, upstream_codes, strict=True):
        result = encoder(frame, *encoder_state)
        explicit_codes.append(result[0])
        encoder_state = result[1:]
        _assert_equal("explicit encoder", result[0], expected)
    encoder_path = args.output_dir / "mimi_streaming_encoder.onnx"
    _export(encoder, audio_frames[0], encoder_initial, encoder_path, "audio", "codes")
    mimi._stop_streaming()

    decoder = ExplicitMimiDecoder(mimi)
    decoder_initial = decoder.initial_state()
    decoder_state = tuple(value.clone() for value in decoder_initial)
    explicit_audio = []
    for codes, expected in zip(upstream_codes, upstream_audio, strict=True):
        result = decoder(codes, *decoder_state)
        explicit_audio.append(result[0])
        decoder_state = result[1:]
        _assert_equal("explicit decoder", result[0], expected)
    decoder_path = args.output_dir / "mimi_streaming_decoder.onnx"
    _export(decoder, upstream_codes[0], decoder_initial, decoder_path, "codes", "audio")

    ort_codes, encoder_ort_state = _ort_chain(
        encoder_path,
        "audio",
        "codes",
        [value.detach().cpu().numpy() for value in audio_frames],
        encoder_initial,
    )
    for frame, (actual, expected) in enumerate(
        zip(ort_codes, upstream_codes, strict=True)
    ):
        expected_array = expected.detach().cpu().numpy()
        if not np.array_equal(actual, expected_array):
            raise RuntimeError(f"ORT encoder frame {frame} token mismatch")

    ort_audio, decoder_ort_state = _ort_chain(
        decoder_path,
        "codes",
        [value.detach().cpu().numpy() for value in upstream_codes],
        decoder_initial,
    )
    decoder_metrics = []
    for frame, (actual, expected) in enumerate(
        zip(ort_audio, upstream_audio, strict=True)
    ):
        reference = expected.detach().cpu().numpy()
        delta = actual.astype(np.float64) - reference.astype(np.float64)
        rmse = float(np.sqrt(np.mean(delta**2)))
        reference_rms = float(np.sqrt(np.mean(reference.astype(np.float64) ** 2)))
        item = {
            "frame": frame,
            "finite": bool(np.isfinite(actual).all()),
            "max_abs": float(np.max(np.abs(delta))),
            "rmse": rmse,
            "relative_rms": rmse / max(reference_rms, 1e-12),
        }
        decoder_metrics.append(item)
        print(f"ORT decoder frame={frame}: {item}", flush=True)
        if not item["finite"] or item["rmse"] > 0.01 or item["max_abs"] > 0.1:
            raise RuntimeError(f"ORT decoder frame {frame} failed the float candidate gate")

    upstream_wave = np.concatenate(
        [value.detach().cpu().numpy() for value in upstream_audio], axis=-1
    )
    ort_wave = np.concatenate(ort_audio, axis=-1)
    _write_wav(args.output_dir / "decoder_upstream_streaming.wav", upstream_wave)
    _write_wav(args.output_dir / "decoder_onnx_streaming.wav", ort_wave)
    manifest = {
        "format": FORMAT,
        "scope": "two-frame streaming parity with explicit fixed-shape tensor state",
        "frames_verified": 2,
        "frame_samples": FRAME_SAMPLES,
        "host_contract": [
            "initialize position, kv_cache, and conv_state to zero",
            "feed every state output into the same component on the next frame",
            "keep encoder and decoder state independent",
            "reset all three tensors together when starting a new stream",
        ],
        "encoder": {
            "onnx": encoder_path.name,
            "state_spec": asdict(encoder.state_spec),
            "tokens": "exact PASS for both frames",
            "final_state_shapes": [list(value.shape) for value in encoder_ort_state],
        },
        "decoder": {
            "onnx": decoder_path.name,
            "state_spec": asdict(decoder.state_spec),
            "metrics": decoder_metrics,
            "final_state_shapes": [list(value.shape) for value in decoder_ort_state],
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)
    print("Mimi explicit-state PyTorch/ONNX two-frame parity: PASS", flush=True)


if __name__ == "__main__":
    main()

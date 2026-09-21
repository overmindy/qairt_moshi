"""Compare a phone Mimi QNN loopback with the real streaming checkpoint.

The phone CSV contains one frame index followed by eight QNN encoder codes.
The phone WAV is the decoder output from those exact codes. Keep the two error
sources separate: code selection and decoder numerical error for fixed codes.
"""

from __future__ import annotations

import argparse
import csv
import json
import wave
from pathlib import Path

import numpy as np
import torch

from moshi_export_mimi_streaming_onnx import FRAME_SAMPLES, _load_audio, _write_wav
from moshi_probe_mimi_streaming_cloud import _metrics, _reference, _session
from qai_hub_models.models.templates.moshi.external_repos.moshi.moshi.moshi.utils.compile import (
    no_compile,
)
from qai_hub_models.models.templates.moshi.model import resolve_moshi_checkpoint


def read_wav(path: Path, frames: int) -> np.ndarray:
    with wave.open(str(path), "rb") as source:
        if (source.getnchannels(), source.getsampwidth(), source.getframerate()) != (
            1, 2, 24_000,
        ):
            raise ValueError("Expected mono 24 kHz PCM16 WAV")
        if source.getnframes() != frames * FRAME_SAMPLES:
            raise ValueError("Phone WAV does not contain the requested frame count")
        return (np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
                .astype(np.float32).reshape(1, 1, -1) / 32768)


def initial_state(spec: dict) -> list[np.ndarray]:
    return [
        np.zeros(1, np.int64),
        np.zeros(spec["kv_cache_shape"], np.float32),
        np.zeros((1, sum(np.prod(shape) for shape in spec["conv_state_shapes"])),
                 np.float32),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--onnx-dir", type=Path, required=True)
    parser.add_argument("--audio-wav", type=Path, required=True)
    parser.add_argument("--phone-wav", type=Path, required=True)
    parser.add_argument("--phone-codes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--torch-device", default="cpu")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.phone_codes.open(newline="") as source:
        rows = [[int(value) for value in row] for row in csv.reader(source)]
    if len(rows) != args.frames or any(row[0] != index or len(row) != 9
                                    for index, row in enumerate(rows)):
        raise ValueError("Phone CSV must contain consecutive frames and eight codes each")
    phone_codes = [np.asarray(row[1:], dtype=np.int32).reshape(1, 8, 1)
                   for row in rows]
    phone_audio = read_wav(args.phone_wav, args.frames)
    source_audio = _load_audio(args.audio_wav, args.frames).numpy()
    frames = [source_audio[..., i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES]
              for i in range(args.frames)]
    manifest = json.loads((args.onnx_dir / "manifest.json").read_text())

    encoder_ort = _reference(
        _session(args.onnx_dir / manifest["encoder"]["onnx"]), frames, "audio",
        initial_state(manifest["encoder"]["state_spec"]),
    )
    decoder_graph = args.onnx_dir / manifest["decoder"]["onnx"]
    decoder_ort_phone_codes = _reference(
        _session(decoder_graph), phone_codes, "codes",
        initial_state(manifest["decoder"]["state_spec"]),
    )
    decoder_ort_reference_codes = _reference(
        _session(decoder_graph), [item[0] for item in encoder_ort], "codes",
        initial_state(manifest["decoder"]["state_spec"]),
    )

    mimi = resolve_moshi_checkpoint(model_dir=args.model_dir).get_mimi(
        device=args.torch_device
    )
    upstream_codes = []
    upstream_audio = []
    with torch.no_grad(), no_compile(), mimi.streaming(1):
        for frame in frames:
            codes = mimi.encode(torch.from_numpy(frame).to(args.torch_device))
            upstream_codes.append(codes.to(torch.int32).cpu().numpy())
            upstream_audio.append(mimi.decode(codes)[..., :FRAME_SAMPLES].cpu().numpy())

    upstream_wav = np.concatenate(upstream_audio, axis=-1)
    ort_phone_wav = np.concatenate([item[0] for item in decoder_ort_phone_codes],
                                   axis=-1)
    ort_reference_wav = np.concatenate(
        [item[0] for item in decoder_ort_reference_codes], axis=-1
    )
    token_metrics = [_metrics(phone, reference) for phone, reference in
                     zip(phone_codes, upstream_codes, strict=True)]
    report = {
        "frames": args.frames,
        "phone_codes_vs_pytorch": token_metrics,
        "phone_matching_tokens": sum(item["matching"] for item in token_metrics),
        "total_tokens": args.frames * 8,
        "ort_codes_vs_pytorch": [_metrics(item[0], reference) for item, reference in
                                 zip(encoder_ort, upstream_codes, strict=True)],
        "phone_audio_vs_pytorch": _metrics(phone_audio, upstream_wav),
        "ort_phone_codes_vs_pytorch": _metrics(ort_phone_wav, upstream_wav),
        "phone_audio_vs_ort_same_codes": _metrics(phone_audio, ort_phone_wav),
        "ort_reference_codes_vs_pytorch": _metrics(ort_reference_wav,
                                                     upstream_wav),
        "per_frame_phone_audio_vs_pytorch": [
            _metrics(phone_audio[..., i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES],
                     upstream_audio[i]) for i in range(args.frames)
        ],
    }
    (args.output_dir / f"phone_mimi_{args.frames}_frames.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    _write_wav(args.output_dir / f"pytorch_mimi_{args.frames}_frames.wav",
               upstream_wav)
    _write_wav(args.output_dir / f"ort_phone_codes_{args.frames}_frames.wav",
               ort_phone_wav)
    print(json.dumps({key: value for key, value in report.items() if key in (
        "frames", "phone_matching_tokens", "total_tokens",
        "phone_audio_vs_pytorch", "phone_audio_vs_ort_same_codes",
    )}, indent=2), flush=True)


if __name__ == "__main__":
    main()

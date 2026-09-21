"""Assess the audio effect of Mimi QNN encoder code differences.

Replay QNN encoder codes through the stateful ONNX decoder and compare with
the same decoder fed ONNX encoder codes. This isolates encoder token changes
without submitting another cloud job.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from moshi_export_mimi_streaming_onnx import FRAME_SAMPLES, _load_audio, _write_wav
from moshi_probe_mimi_streaming_cloud import _metrics, _reference, _session


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-dir", type=Path, required=True)
    parser.add_argument("--audio-wav", type=Path, required=True)
    parser.add_argument("--qnn-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=13)
    args = parser.parse_args()

    manifest = json.loads((args.onnx_dir / "manifest.json").read_text())
    audio = _load_audio(args.audio_wav, args.frames).numpy()
    frames = [audio[..., i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES]
              for i in range(args.frames)]
    encoder_spec = manifest["encoder"]["state_spec"]
    encoder_initial = [
        np.zeros(1, np.int64),
        np.zeros(encoder_spec["kv_cache_shape"], np.float32),
        np.zeros((1, sum(np.prod(shape) for shape in
                          encoder_spec["conv_state_shapes"])), np.float32),
    ]
    encoder_reference = _reference(
        _session(args.onnx_dir / manifest["encoder"]["onnx"]),
        frames, "audio", encoder_initial,
    )
    qnn_codes = []
    for frame in range(args.frames):
        with np.load(args.qnn_dir / f"encoder_qnn_frame_{frame}.npz") as archive:
            qnn_codes.append(archive["output_0"].copy())
    decoder_spec = manifest["decoder"]["state_spec"]
    decoder_initial = [
        np.zeros(1, np.int64),
        np.zeros(decoder_spec["kv_cache_shape"], np.float32),
        np.zeros((1, sum(np.prod(shape) for shape in
                          decoder_spec["conv_state_shapes"])), np.float32),
    ]
    decoder_graph = args.onnx_dir / manifest["decoder"]["onnx"]
    reference = _reference(
        _session(decoder_graph), [item[0] for item in encoder_reference],
        "codes", decoder_initial,
    )
    actual = _reference(_session(decoder_graph), qnn_codes,
                        "codes", decoder_initial)
    token_metrics = [_metrics(code, ref[0]) for code, ref in
                     zip(qnn_codes, encoder_reference, strict=True)]
    waveform_metrics = [_metrics(output[0], ref[0]) for output, ref in
                        zip(actual, reference, strict=True)]
    waveform = np.concatenate([item[0] for item in actual], axis=-1)
    baseline = np.concatenate([item[0] for item in reference], axis=-1)
    aggregate = _metrics(waveform, baseline)
    report = {
        "scope": "ONNX decoder with QNN encoder codes versus ONNX encoder codes",
        "frames": args.frames,
        "matching_tokens": sum(item["matching"] for item in token_metrics),
        "total_tokens": sum(item["total"] for item in token_metrics),
        "token_metrics": token_metrics,
        "waveform_metrics": waveform_metrics,
        "aggregate_waveform": aggregate,
    }
    (args.qnn_dir / f"qnn_code_audio_assessment_{args.frames}_frames.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    _write_wav(args.qnn_dir / f"decoder_ort_from_qnn_codes_{args.frames}_frames.wav",
               waveform)
    print(json.dumps({"matching_tokens": report["matching_tokens"],
                      "total_tokens": report["total_tokens"],
                      "aggregate_waveform": aggregate}, indent=2))


if __name__ == "__main__":
    main()

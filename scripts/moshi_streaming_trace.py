"""Run the real Kyutai Moshi streaming path and save a golden trace."""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qai_hub_models.models.moshi.model import Moshi
from qai_hub_models.models.templates.moshi.app import MoshiStreamingApp


@torch.no_grad()
def verify_temporal_replay(lm, trace: dict[str, torch.Tensor]) -> None:
    sequence = trace["temporal_sequence"]
    failures = []
    with lm.streaming(sequence.shape[0]):
        for frame in range(sequence.shape[-1]):
            hidden, logits = lm.forward_text(sequence[..., frame:frame + 1].to(lm.device))
            for name, actual, expected in (
                ("temporal_hidden", hidden, trace["temporal_hidden"][:, frame:frame + 1]),
                ("text_logits", logits, trace["text_logits"][:, :, frame:frame + 1]),
            ):
                actual = actual.detach().cpu()
                if actual.shape != expected.shape or actual.dtype != expected.dtype:
                    raise RuntimeError(f"frame={frame} {name}: shape or dtype mismatch")
                exact = torch.equal(actual, expected)
                maximum = (actual.float() - expected.float()).abs().max().item()
                print(f"replay frame={frame} {name}: exact={exact} max_abs={maximum:.8g}")
                if not exact or not torch.isfinite(actual).all():
                    failures.append((frame, name))
    if failures:
        raise RuntimeError(f"Temporal replay differed from golden trace: {failures}")
    print("Temporal replay: PASS (fresh streaming state, all frames exact)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--verify-temporal-replay", action="store_true")
    args = parser.parse_args()
    if args.frames < 1:
        raise SystemExit("--frames must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but torch.cuda.is_available() is false.")

    model = Moshi.from_pretrained(model_dir=args.model_dir, device=args.device)
    streaming = MoshiStreamingApp(
        model.components["encoder"].model,
        model.components["temporal"].model,
        use_sampling=False,
    )
    audio = torch.zeros(1, 1, args.frames * 1920, device=args.device)
    if args.frames <= streaming.generator.max_delay:
        raise SystemExit(f"--frames must exceed LMGen max_delay={streaming.generator.max_delay}")
    waveform, trace = streaming.run(audio)
    if not waveform.numel() or not torch.isfinite(waveform).all():
        raise RuntimeError("Streaming returned empty or non-finite audio")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(audio.cpu(), args.output_dir / "input_audio.pt")
    for name, value in trace.items():
        torch.save(value, args.output_dir / f"{name}.pt")
    pcm = waveform[0].clamp(-1, 1).mul(32767).to(torch.int16).numpy().astype("<i2")
    with wave.open(str(args.output_dir / "waveform.wav"), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24000)
        output.writeframes(pcm.tobytes())
    print(f"frames={args.frames} output_dir={args.output_dir}")
    for name, value in trace.items():
        print(f"{name}_shape={tuple(value.shape)} dtype={value.dtype}")
    if args.verify_temporal_replay:
        verify_temporal_replay(model.components["temporal"].model, trace)


if __name__ == "__main__":
    main()

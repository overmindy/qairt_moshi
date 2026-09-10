"""Prepare real replay embeddings for existing compiled Temporal shards without re-export."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")
    if args.output.suffix != ".npy" or args.output.exists():
        parser.error("--output must be a new .npy file")

    import numpy as np
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from qai_hub_models.models.moshi.model import Moshi
    from qai_hub_models.models.templates.moshi.explicit_cache import ExplicitTemporal
    from qai_hub_models.models.templates.moshi.external_repos.moshi.moshi.moshi.utils.compile import no_compile

    sequence_path = args.trace_dir / "temporal_sequence.pt"
    sequence = torch.load(sequence_path, map_location="cpu", weights_only=True)
    if sequence.ndim != 3 or sequence.shape[0] != 1 or sequence.shape[-1] < args.frames:
        raise ValueError(f"Need at least {args.frames} saved frames, got {tuple(sequence.shape)}")
    model = Moshi.from_pretrained(model_dir=args.model_dir, device=args.device)
    temporal = ExplicitTemporal(model.components["temporal"].model).eval()
    embeddings = []
    with torch.no_grad(), no_compile():
        for frame in range(args.frames):
            hidden = temporal.embed(sequence[..., frame:frame + 1].to(args.device))
            embeddings.append(hidden.cpu().float().numpy().copy())
    values = np.stack(embeddings)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite embeddings")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as output:
        np.save(output, values)
    args.output.with_suffix(".json").write_text(json.dumps({
        "model_dir": str(args.model_dir.resolve()),
        "sequence": str(sequence_path.resolve()), "shape": list(values.shape),
        "precision": "BF16 embedding operations, saved as FP32",
        "scope": "Saved replay inputs; no cloud prediction feedback or new ONNX export",
    }, indent=2) + "\n")
    print(f"Saved {args.output}: shape={values.shape}; no cloud jobs submitted")


if __name__ == "__main__":
    main()

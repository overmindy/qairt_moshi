"""Run Moshi checks with an explicit opt-in for loading real weights."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from qai_hub_models.models.moshi import Model
from qai_hub_models.models.templates.moshi.app import MoshiApp


def _static_check() -> None:
    required = [
        Path("src/qai_hub_models/models/moshi/manifest.yaml"),
        Path("src/qai_hub_models/models/moshi/requirements.txt"),
        Path("src/qai_hub_models/models/moshi/model.py"),
        Path("src/qai_hub_models/models/templates/moshi/model.py"),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing Moshi files: {', '.join(missing)}")
    print("Static Moshi recipe check passed.")


def _real_smoke(hf_repo: str) -> None:
    print(f"Loading one real Moshi checkpoint: {hf_repo}")
    with torch.inference_mode():
        model = Model.from_pretrained(hf_repo=hf_repo)
        app = MoshiApp(model.encoder, model.temporal, model.depformer, model.decoder)
        audio = torch.zeros(1, 1, 1920)
        waveform, text_logits = app(audio)
    print(f"waveform_shape={tuple(waveform.shape)} text_logits_shape={tuple(text_logits.shape)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-real", action="store_true", help="Load the real Moshi checkpoint and run one frame.")
    parser.add_argument("--hf-repo", default="kyutai/moshiko-pytorch-bf16")
    args = parser.parse_args()
    _static_check()
    if args.run_real:
        _real_smoke(args.hf_repo)
    else:
        print("Real weights were not loaded. Pass --run-real on a memory-capable Linux host.")


if __name__ == "__main__":
    main()

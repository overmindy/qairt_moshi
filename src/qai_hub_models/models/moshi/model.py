# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path

import torch
from typing_extensions import Self

from qai_hub_models.models.templates.moshi.model import (
    MimiDecoder,
    MimiEncoder,
    MoshiDepFormer,
    MoshiTemporal,
    load_moshi_models,
)
from qai_hub_models.utils.base_collection_model import WorkbenchModelCollection

MODEL_ID = __name__.split(".")[-2]


class Moshi(WorkbenchModelCollection):
    def __init__(
        self,
        encoder: MimiEncoder,
        temporal: MoshiTemporal,
        depformer: MoshiDepFormer,
        decoder: MimiDecoder,
    ) -> None:
        super().__init__({"encoder": encoder, "temporal": temporal, "depformer": depformer, "decoder": decoder})

    @classmethod
    def from_pretrained(
        cls,
        hf_repo: str = "kyutai/moshiko-pytorch-bf16",
        model_dir: str | Path | None = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ) -> Self:
        mimi_encoder, lm, mimi_decoder = load_moshi_models(
            hf_repo=hf_repo,
            model_dir=model_dir,
            device=device,
            dtype=dtype,
        )
        return cls(
            MimiEncoder(mimi_encoder),
            MoshiTemporal(lm),
            MoshiDepFormer(lm),
            MimiDecoder(mimi_decoder),
        )

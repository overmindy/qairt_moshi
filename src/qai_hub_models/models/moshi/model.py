# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

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
    def from_pretrained(cls, hf_repo: str | None = None) -> Self:
        mimi_encoder, lm, mimi_decoder = load_moshi_models(
            hf_repo or "kyutai/moshiko-pytorch-bf16"
        )
        return cls(
            MimiEncoder(mimi_encoder),
            MoshiTemporal(lm),
            MoshiDepFormer(lm),
            MimiDecoder(mimi_decoder),
        )

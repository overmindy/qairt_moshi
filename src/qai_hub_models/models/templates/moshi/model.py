# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch
from torch import nn
from typing_extensions import Self

from qai_hub_models.models.templates.moshi.external_repos.moshi.moshi.moshi.models.compression import (
    MimiModel,
)
from qai_hub_models.models.templates.moshi.external_repos.moshi.moshi.moshi.models.lm import (
    LMModel,
)
from qai_hub_models.models.templates.moshi.external_repos.moshi.moshi.moshi.models.loaders import (
    CheckpointInfo,
    get_mimi,
    get_moshi_lm,
)
from qai_hub_models.utils.base_model import PytorchWorkbenchModel, SerializationSettings
from qai_hub_models.utils.input_spec import InputSpec, OutputSpec, TensorSpec

FRAME_SAMPLES = 1920
CODEBOOKS = 8
DEFAULT_HF_REPO = "kyutai/moshiko-pytorch-bf16"


class _StaticComponent(PytorchWorkbenchModel):
    def __init__(self, module: nn.Module) -> None:
        super().__init__(module, SerializationSettings(use_pt2=False, check_trace=False))


class MimiEncoder(_StaticComponent):
    def __init__(self, module: MimiModel) -> None:
        super().__init__(module)

    @classmethod
    def from_pretrained(cls, checkpoint: str | None = None) -> Self:
        return cls(get_mimi(checkpoint, num_codebooks=CODEBOOKS))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return self.model.encode(audio).to(torch.int32)

    def get_input_spec(self) -> InputSpec:
        return {"audio": TensorSpec((1, 1, FRAME_SAMPLES), "float32")}

    def get_output_spec(self) -> OutputSpec:
        return {"codes": TensorSpec((1, self.model.num_codebooks, 1), "int32")}


class MoshiTemporal(_StaticComponent):
    def __init__(self, module: LMModel) -> None:
        super().__init__(module)

    @classmethod
    def from_pretrained(cls, checkpoint: str | None = None) -> Self:
        return cls(get_moshi_lm(checkpoint, dtype=torch.bfloat16))

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.forward_text(sequence)

    def get_input_spec(self) -> InputSpec:
        return {"sequence": TensorSpec((1, self.model.num_codebooks, 1), "int32")}

    def get_output_spec(self) -> OutputSpec:
        return {
            "temporal": TensorSpec((1, 1, self.model.dim), "float32"),
            "text_logits": TensorSpec((1, 1, 1, self.model.text_linear.out_features), "float32"),
        }


class MoshiDepFormer(_StaticComponent):
    def __init__(self, module: LMModel) -> None:
        super().__init__(module)

    @classmethod
    def from_pretrained(cls, checkpoint: str | None = None) -> Self:
        return cls(get_moshi_lm(checkpoint, dtype=torch.bfloat16))

    def forward(self, sequence: torch.Tensor, temporal: torch.Tensor) -> torch.Tensor:
        return self.model.forward_depformer_training(sequence, temporal)

    def get_input_spec(self) -> InputSpec:
        return {
            "sequence": TensorSpec((1, self.model.num_codebooks, 1), "int32"),
            "temporal": TensorSpec((1, 1, self.model.dim), "float32"),
        }

    def get_output_spec(self) -> OutputSpec:
        return {"logits": TensorSpec((1, self.model.dep_q, 1, self.model.card), "float32")}


class MimiDecoder(_StaticComponent):
    def __init__(self, module: MimiModel) -> None:
        super().__init__(module)

    @classmethod
    def from_pretrained(cls, checkpoint: str | None = None) -> Self:
        return cls(get_mimi(checkpoint, num_codebooks=CODEBOOKS))

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        return self.model.decode(codes.long())[..., :FRAME_SAMPLES]

    def get_input_spec(self) -> InputSpec:
        return {"codes": TensorSpec((1, self.model.num_codebooks, 1), "int32")}

    def get_output_spec(self) -> OutputSpec:
        return {"audio": TensorSpec((1, 1, FRAME_SAMPLES), "float32")}


def load_moshi_models(
    hf_repo: str = DEFAULT_HF_REPO,
) -> tuple[MimiModel, LMModel, MimiModel]:
    checkpoint = CheckpointInfo.from_hf_repo(hf_repo)
    mimi = checkpoint.get_mimi()
    return mimi, checkpoint.get_moshi(dtype=torch.bfloat16), mimi

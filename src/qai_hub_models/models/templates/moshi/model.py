# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import json
from pathlib import Path

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
MODEL_WEIGHTS_NAME = "model.safetensors"
MIMI_WEIGHTS_NAME = "tokenizer-e351c8d8-checkpoint125.safetensors"
TOKENIZER_NAME = "tokenizer_spm_32k_3.model"


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
    model_dir: str | Path | None = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[MimiModel, LMModel, MimiModel]:
    if model_dir is None:
        checkpoint = CheckpointInfo.from_hf_repo(hf_repo)
    else:
        model_path = Path(model_dir).expanduser().resolve()
        if not model_path.is_dir():
            raise FileNotFoundError(f"Moshi model directory does not exist: {model_path}")
        config_path = model_path / "config.json"
        config = json.loads(config_path.read_text()) if config_path.is_file() else {}
        moshi_weights = model_path / config.get("moshi_name", MODEL_WEIGHTS_NAME)
        mimi_weights = model_path / config.get("mimi_name", MIMI_WEIGHTS_NAME)
        tokenizer = model_path / config.get("tokenizer_name", TOKENIZER_NAME)
        required_paths = [moshi_weights, mimi_weights, tokenizer]
        missing = [str(path) for path in required_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing files in local Moshi directory: {', '.join(missing)}")
        if not config_path.is_file():
            checkpoint = CheckpointInfo(moshi_weights, mimi_weights, tokenizer)
        else:
            mimi_config_name = config.get("mimi_config_name")
            mimi_config_path = model_path / mimi_config_name if mimi_config_name else None
            lora_name = config.get("lora_name")
            lora_path = model_path / lora_name if lora_name else None
            optional_paths = [path for path in (mimi_config_path, lora_path) if path]
            missing_optional = [str(path) for path in optional_paths if not path.is_file()]
            if missing_optional:
                raise FileNotFoundError(
                    "Missing files referenced by config.json: "
                    + ", ".join(missing_optional)
                )
            checkpoint = CheckpointInfo.from_hf_repo(
                hf_repo,
                moshi_weights=moshi_weights,
                mimi_weights=mimi_weights,
                tokenizer=tokenizer,
                config_path=config_path,
                mimi_config_path=mimi_config_path,
                lora_weights=lora_path,
            )
    mimi = checkpoint.get_mimi(device=device)
    return mimi, checkpoint.get_moshi(device=device, dtype=dtype), mimi

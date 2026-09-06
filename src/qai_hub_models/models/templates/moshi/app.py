# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from collections.abc import Callable

import torch


class MoshiApp:
    """Offline one-frame pipeline with explicit codec/model boundaries."""

    def __init__(
        self,
        encoder: Callable,
        temporal: Callable,
        depformer: Callable,
        decoder: Callable,
    ) -> None:
        self.encoder = encoder
        self.temporal = temporal
        self.depformer = depformer
        self.decoder = decoder

    def predict(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        user_codes = self.encoder(audio)
        text = torch.full(
            (audio.shape[0], 1, 1),
            3,
            dtype=torch.int32,
            device=audio.device,
        )
        response_context = torch.zeros_like(user_codes)
        sequence = torch.cat((text, response_context, user_codes), dim=1)
        temporal, text_logits = self.temporal(sequence)
        logits = self.depformer(sequence, temporal)
        codes = logits.argmax(dim=-1).to(torch.int32)
        waveform = self.decoder(codes)
        return waveform, text_logits

    __call__ = predict

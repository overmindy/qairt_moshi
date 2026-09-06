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


class MoshiStreamingApp:
    """Kyutai LMGen-compatible multi-frame streaming driver."""

    def __init__(self, mimi, lm, *, use_sampling: bool = False) -> None:
        self.mimi = mimi
        from .external_repos.moshi.moshi.moshi.models.lm import LMGen
        self._text_logits = []
        self._depformer_logits = []
        self.generator = LMGen(
            lm, use_sampling=use_sampling,
            on_text_logits_hook=lambda value: self._text_logits.append(value.detach().cpu()),
            on_depformer_logits_hook=lambda value: self._depformer_logits.append(value.detach().cpu()),
        )

    @torch.no_grad()
    def run(self, audio: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if audio.ndim != 3 or audio.shape[1] != 1 or audio.shape[-1] % 1920:
            raise ValueError("audio must be [B, 1, frames*1920]")
        codes = self.mimi.encode(audio)
        self._text_logits.clear()
        self._depformer_logits.clear()
        traces: dict[str, list[torch.Tensor]] = {"mimi_codes": [], "audio_codes": []}
        traces["temporal_hidden"] = []
        traces["text_logits"] = []
        traces["depformer_logits"] = []
        waves = []
        with self.generator.streaming(audio.shape[0]):
            for frame in range(codes.shape[-1]):
                user = codes[..., frame:frame + 1]
                # LMGen accepts the user codebooks and owns generated streams/cache.
                result, hidden = self.generator._step(user)
                traces["mimi_codes"].append(user.detach().cpu())
                if result is None:
                    continue
                traces["audio_codes"].append(result.detach().cpu())
                traces["temporal_hidden"].append(hidden.detach().cpu())
                waves.append(self.mimi.decode(result.to(codes.device)).detach().cpu())
        def cat(name: str) -> torch.Tensor:
            values = traces[name]
            return torch.cat(values, dim=-1) if values else torch.empty(0)
        trace = {name: cat(name) for name in traces}
        trace["waveform"] = torch.cat(waves, dim=-1) if waves else torch.empty(0)
        trace["text_logits"] = torch.cat(self._text_logits, dim=2) if self._text_logits else torch.empty(0)
        trace["depformer_logits"] = torch.stack(self._depformer_logits, dim=2) if self._depformer_logits else torch.empty(0)
        return trace["waveform"], trace

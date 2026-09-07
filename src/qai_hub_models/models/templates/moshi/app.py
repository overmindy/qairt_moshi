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

        mimi_frames = []
        audio_frames = []
        temporal_frames = []
        waves = []

        with self.generator.streaming(audio.shape[0]):
            for frame in range(codes.shape[-1]):
                user = codes[..., frame:frame + 1]
                step_result = self.generator._step(user)
                mimi_frames.append(user.detach().cpu())

                if step_result is None:
                    continue

                result, hidden = step_result
                audio_frames.append(result.detach().cpu())
                temporal_frames.append(hidden.detach().cpu())
                waves.append(
                    self.mimi.decode(result.to(codes.device)).detach().cpu()
                )

        def cat_frames(values):
            return torch.cat(values, dim=-1) if values else torch.empty(0)

        trace = {
            "mimi_codes": cat_frames(mimi_frames),
            "audio_codes": cat_frames(audio_frames),
            "temporal_hidden": (
                torch.cat(temporal_frames, dim=1)
                if temporal_frames else torch.empty(0)
            ),
            "text_logits": (
                torch.cat(self._text_logits, dim=2)
                if self._text_logits else torch.empty(0)
            ),
            "depformer_logits": torch.empty(0),
            "waveform": (
                torch.cat(waves, dim=-1)
                if waves else torch.empty(0)
            ),
        }
        if self._depformer_logits:
            depformer = torch.cat(self._depformer_logits, dim=2)
            # The hook fires once per DepFormer codebook. Restore [B,T,Q,1,Card].
            steps = depformer.shape[2]
            codebooks = self.generator.lm_model.dep_q
            if steps % codebooks:
                raise RuntimeError(f"DepFormer trace steps {steps} not divisible by {codebooks}")
            trace["depformer_logits"] = depformer.reshape(
                depformer.shape[0], 1, steps // codebooks, codebooks,
                depformer.shape[3]
            ).permute(0, 2, 3, 1, 4).contiguous()
        return trace["waveform"], trace

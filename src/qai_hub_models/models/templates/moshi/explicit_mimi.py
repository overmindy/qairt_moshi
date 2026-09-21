# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Fixed-shape Mimi frame graphs with explicit streaming tensor state.

The pinned Mimi implementation keeps convolution tails and Transformer caches
inside Python objects.  These adapters expose the state as graph inputs and
outputs while preserving the upstream frame computation.  State *shapes* are
fixed and preallocated; state *values* change on every 80 ms frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Literal

import torch
from torch import nn

from .external_repos.moshi.moshi.moshi.models.compression import MimiModel


@dataclass(frozen=True)
class MimiExplicitStateSpec:
    """Serializable description of one encoder or decoder state contract."""

    component: Literal["encoder", "decoder"]
    conv_state_shapes: tuple[tuple[int, ...], ...]
    conv_state_names: tuple[str, ...]
    first_state_names: tuple[str, ...]
    transformer_layers: int
    kv_cache_shape: tuple[int, ...]

    @property
    def conv_state_numel(self) -> int:
        return sum(prod(shape) for shape in self.conv_state_shapes)

    @property
    def first_state_count(self) -> int:
        return len(self.first_state_names)


class _ExplicitMimiFrame(nn.Module):
    """Common tensor-state plumbing around the pinned upstream Mimi modules."""

    def __init__(self, model: MimiModel, component: Literal["encoder", "decoder"]):
        super().__init__()
        self.model = model
        self.component = component
        if model.is_streaming:
            raise ValueError("Mimi must not already be in streaming mode")
        # Initialize every child state once. The adapters invoke the underlying
        # modules directly, avoiding the CUDA-graph callables in _MimiState.
        model.streaming_forever(1)
        states = model.get_streaming_state()
        prefixes = (
            ("encoder.", "encoder_transformer.", "downsample.")
            if component == "encoder"
            else ("decoder.", "decoder_transformer.", "upsample.")
        )

        conv_entries: list[tuple[str, object, str, tuple[int, ...]]] = []
        first_entries: list[tuple[str, object]] = []
        attention_entries: list[tuple[str, object]] = []
        transformer_offsets = []
        for name, state in states.items():
            if not name.startswith(prefixes):
                continue
            previous = getattr(state, "previous", None)
            if isinstance(previous, torch.Tensor) and previous.numel():
                conv_entries.append((name + ".previous", state, "previous", tuple(previous.shape)))
                if hasattr(state, "first"):
                    first_entries.append((name + ".first", state))
            partial = getattr(state, "partial", None)
            if isinstance(partial, torch.Tensor) and partial.numel():
                conv_entries.append((name + ".partial", state, "partial", tuple(partial.shape)))
            kv_cache = getattr(state, "kv_cache", None)
            if kv_cache is not None:
                attention_entries.append((name, state))
            offsets = getattr(state, "offsets", None)
            if isinstance(offsets, torch.Tensor):
                transformer_offsets.append(state)

        if not attention_entries or len(transformer_offsets) != 1:
            raise RuntimeError(
                f"Unexpected {component} Transformer state: "
                f"attention={len(attention_entries)} offsets={len(transformer_offsets)}"
            )
        cache_shapes = {
            tuple(entry[1].kv_cache.cache.shape) for entry in attention_entries
        }
        if len(cache_shapes) != 1:
            raise RuntimeError(f"Heterogeneous {component} attention caches: {cache_shapes}")

        self._conv_entries = conv_entries
        self._first_entries = first_entries
        self._attention_entries = attention_entries
        self._transformer_offset_state = transformer_offsets[0]
        cache_shape = cache_shapes.pop()
        if cache_shape[0] != 2:
            raise RuntimeError(f"Unexpected {component} K/V axis: {cache_shape}")
        self.state_spec = MimiExplicitStateSpec(
            component=component,
            conv_state_shapes=tuple(entry[3] for entry in conv_entries),
            conv_state_names=tuple(entry[0] for entry in conv_entries),
            first_state_names=tuple(entry[0] for entry in first_entries),
            transformer_layers=len(attention_entries),
            # Flatten the layer and K/V axes: QNN HTP rejects the rank-six
            # Gather created by selecting a layer from [layers, 2, B, H, T, D].
            kv_cache_shape=(len(attention_entries) * cache_shape[0], *cache_shape[1:]),
        )

    def initial_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return position, K/V cache, and convolution state tensors."""
        parameter = next(self.model.parameters())
        device = parameter.device
        dtype = parameter.dtype
        position = torch.zeros(1, dtype=torch.int64, device=device)
        kv_cache = torch.zeros(self.state_spec.kv_cache_shape, dtype=dtype, device=device)
        conv_state = torch.zeros(
            1, self.state_spec.conv_state_numel, dtype=dtype, device=device
        )
        return position, kv_cache, conv_state

    def _install_state(
        self,
        position: torch.Tensor,
        kv_cache: torch.Tensor,
        conv_state: torch.Tensor,
    ) -> None:
        if not torch.jit.is_tracing():
            if position.shape != (1,):
                raise ValueError(f"position must be [1], got {tuple(position.shape)}")
            if tuple(kv_cache.shape) != self.state_spec.kv_cache_shape:
                raise ValueError(
                    f"kv_cache must be {self.state_spec.kv_cache_shape}, "
                    f"got {tuple(kv_cache.shape)}"
                )
            if tuple(conv_state.shape) != (1, self.state_spec.conv_state_numel):
                raise ValueError("conv_state shape does not match the explicit state spec")
        cursor = 0
        for _, state, field, shape in self._conv_entries:
            count = prod(shape)
            value = conv_state[:, cursor : cursor + count].reshape(shape).clone()
            setattr(state, field, value)
            cursor += count
        for _, state in self._first_entries:
            # In a batch-one stream the first-frame flag is exactly position=0.
            # Deriving it removes boolean cache I/O and makes resetting a stream
            # equivalent to supplying the zero initial state again.
            state.first = (position == 0).clone()

        # A valid Mimi stream advances every Transformer position together.
        # Keep a single host-owned position instead of exposing three redundant
        # counters per layer.
        self._transformer_offset_state.offsets = position.clone()
        for layer, (_, state) in enumerate(self._attention_entries):
            state.offset = position.clone()
            # Mimi has one projection per attention, so offset_cpu is not used
            # to select weights. Keeping it at zero avoids a tensor-to-Python
            # conversion in the exported graph.
            state.offset_cpu = 0
            state.kv_cache.cache = kv_cache[layer * 2 : layer * 2 + 2].clone()
            state.kv_cache.end_offset = position.clone()

    def _collect_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # `position` is measured in Transformer timesteps, not codec frames.
        # Mimi's encoder Transformer runs before the 2:1 downsampler and its
        # decoder Transformer runs after the 1:2 upsampler, so one 80 ms codec
        # frame advances this value by two. Read the value updated by the
        # upstream attention instead of duplicating that rate conversion here.
        position_out = self._attention_entries[0][1].offset.clone()
        caches = torch.cat(
            [state.kv_cache.cache for _, state in self._attention_entries], dim=0
        )
        convolution = torch.cat(
            [getattr(state, field).reshape(1, -1) for _, state, field, _ in self._conv_entries],
            dim=1,
        )
        return position_out, caches, convolution


class ExplicitMimiEncoder(_ExplicitMimiFrame):
    """One streaming Mimi encoder frame with explicit preallocated state."""

    def __init__(self, model: MimiModel):
        super().__init__(model, "encoder")

    def forward(
        self,
        audio: torch.Tensor,
        position: torch.Tensor,
        kv_cache: torch.Tensor,
        conv_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._install_state(position, kv_cache, conv_state)
        embedding = self.model.encoder(audio)
        if self.model.encoder_transformer is not None:
            (embedding,) = self.model.encoder_transformer(embedding)
        embedding = self.model._to_framerate(embedding)
        codes = self.model.quantizer.encode(embedding).to(torch.int32)
        return codes, *self._collect_state()


class ExplicitMimiDecoder(_ExplicitMimiFrame):
    """One streaming Mimi decoder frame with explicit preallocated state."""

    def __init__(self, model: MimiModel):
        super().__init__(model, "decoder")

    def forward(
        self,
        codes: torch.Tensor,
        position: torch.Tensor,
        kv_cache: torch.Tensor,
        conv_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._install_state(position, kv_cache, conv_state)
        embedding = self.model.decode_latent(codes.long())
        embedding = self.model._to_encoder_framerate(embedding)
        if self.model.decoder_transformer is not None:
            (embedding,) = self.model.decoder_transformer(embedding)
        audio = self.model.decoder(embedding)
        return audio[..., :1920], *self._collect_state()

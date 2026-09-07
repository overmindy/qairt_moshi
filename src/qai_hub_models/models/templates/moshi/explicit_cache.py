"""Functional single-step Temporal blocks with explicit ring KV tensors."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as functional


class TemporalBlock(nn.Module):
    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        attention = layer.self_attn
        if layer.skip_self_attn or layer.cross_attention is not None or layer.weights_per_step:
            raise ValueError("Only unconditional Temporal self-attention blocks are supported")
        if not attention.causal or attention.context is None:
            raise ValueError("A finite causal attention context is required")
        if len(attention.in_projs) != 1 or len(attention.out_projs) != 1:
            raise ValueError("Per-step attention projections are unsupported")
        self.layer = layer
        self.capacity = attention.context
        self.heads = attention.num_heads
        self.kv_heads = attention.num_heads // attention.kv_repeat
        self.head_dim = attention.embed_dim // attention.num_heads

    def empty_cache(self) -> torch.Tensor:
        parameter = next(self.parameters())
        return parameter.new_zeros((2, 1, self.kv_heads, self.capacity, self.head_dim))

    def forward(
        self, hidden: torch.Tensor, cache: torch.Tensor, position: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attention = self.layer.self_attn
        projected = attention.in_projs[0](self.layer.norm1(hidden))
        query_width = self.heads * self.head_dim
        kv_width = self.kv_heads * self.head_dim
        query, key, value = projected.split((query_width, kv_width, kv_width), dim=-1)
        query = query.reshape(1, 1, self.heads, self.head_dim).transpose(1, 2)
        key = key.reshape(1, 1, self.kv_heads, self.head_dim).transpose(1, 2)
        value = value.reshape(1, 1, self.kv_heads, self.head_dim).transpose(1, 2)
        if attention.rope is not None:
            query, key = attention.rope(query, key, position, time_before_heads=False)
        indexes = (position % self.capacity).reshape(1, 1, 1, 1).expand_as(key)
        keys = cache[0].scatter(2, indexes, key)
        values = cache[1].scatter(2, indexes, value)
        updated_cache = torch.stack((keys, values))
        slots = torch.arange(self.capacity, device=position.device)
        delta = slots - (position.reshape(1, 1) % self.capacity)
        key_positions = torch.where(delta <= 0, position + delta, position + delta - self.capacity)
        valid = (key_positions >= 0) & ((position - key_positions) < self.capacity)
        if attention.kv_repeat > 1:
            keys = keys.repeat_interleave(attention.kv_repeat, dim=1)
            values = values.repeat_interleave(attention.kv_repeat, dim=1)
        update = functional.scaled_dot_product_attention(
            query, keys, values, valid.reshape(1, 1, 1, self.capacity), dropout_p=0.0
        )
        update = attention.out_projs[0](update.transpose(1, 2).reshape(1, 1, query_width))
        hidden = hidden.to(update) + self.layer.layer_scale_1(update)
        normalized = self.layer.norm2(hidden)
        if self.layer.gating is None:
            update = self.layer.linear2(self.layer.activation(self.layer.linear1(normalized)))
        else:
            update = self.layer.gating(normalized)
        return hidden.to(update) + self.layer.layer_scale_2(update), updated_cache


class ExplicitTemporal(nn.Module):
    def __init__(self, lm: nn.Module) -> None:
        super().__init__()
        if lm.transformer.positional_embedding not in ("rope", "rope_concat", "none"):
            raise ValueError("Only rotary or no positional embeddings are supported")
        self.audio_embeddings = lm.emb
        self.text_embedding = lm.text_emb
        self.audio_offset = lm.audio_offset
        self.blocks = nn.ModuleList(TemporalBlock(layer) for layer in lm.transformer.layers)
        self.out_norm = lm.out_norm
        self.text_linear = lm.text_linear

    def embed(self, sequence: torch.Tensor) -> torch.Tensor:
        hidden = self.audio_embeddings[0](sequence[:, self.audio_offset])
        for index in range(1, len(self.audio_embeddings)):
            hidden = hidden + self.audio_embeddings[index](sequence[:, index + self.audio_offset])
        return hidden + self.text_embedding(sequence[:, 0])

    def forward(
        self, sequence: torch.Tensor, position: torch.Tensor, *caches: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        hidden = self.embed(sequence)
        updated = []
        for block, cache in zip(self.blocks, caches, strict=True):
            hidden, cache = block(hidden, cache, position)
            updated.append(cache)
        if self.out_norm is not None:
            hidden = self.out_norm(hidden)
        return hidden, self.text_linear(hidden)[:, None], position + 1, *updated

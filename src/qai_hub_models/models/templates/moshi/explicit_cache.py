"""Pure inference graphs for Moshi's Temporal and DepFormer transformers.

The upstream implementation stores KV tensors in Python streaming-state objects.
Those objects and the codebook loop in ``LMGen`` are intentionally kept outside
of ONNX.  The modules below expose Temporal state as tensor inputs/outputs and
statically unroll the eight greedy DepFormer steps inside one graph.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as functional


def _weight_index(module: nn.Module, step: int) -> int:
    """Resolve an upstream weights-per-step schedule at export time."""
    schedule = module.weights_per_step_schedule
    return schedule[step] if schedule is not None else step


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
        self.dynamic_rmsnorm = False
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
        hidden, keys, values = self.forward_split(hidden, cache[0], cache[1], position)
        return hidden, torch.stack((keys, values))

    def forward_split(
        self, hidden: torch.Tensor, key_cache: torch.Tensor,
        value_cache: torch.Tensor, position: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        position = position.to(torch.long)
        attention = self.layer.self_attn
        projected = attention.in_projs[0](self.normalize(self.layer.norm1, hidden))
        query_width = self.heads * self.head_dim
        kv_width = self.kv_heads * self.head_dim
        query, key, value = projected.split((query_width, kv_width, kv_width), dim=-1)
        query = query.reshape(1, 1, self.heads, self.head_dim).transpose(1, 2)
        key = key.reshape(1, 1, self.kv_heads, self.head_dim).transpose(1, 2)
        value = value.reshape(1, 1, self.kv_heads, self.head_dim).transpose(1, 2)
        if attention.rope is not None:
            query, key = attention.rope(query, key, position, time_before_heads=False)
        slots = torch.arange(self.capacity, device=position.device)
        write_mask = (slots == position % self.capacity).reshape(1, 1, self.capacity, 1)
        keys = torch.where(write_mask, key, key_cache)
        values = torch.where(write_mask, value, value_cache)
        updated_keys, updated_values = keys, values
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
        normalized = self.normalize(self.layer.norm2, hidden)
        if self.layer.gating is None:
            update = self.layer.linear2(self.layer.activation(self.layer.linear1(normalized)))
        else:
            update = self.layer.gating(normalized)
        return hidden.to(update) + self.layer.layer_scale_2(update), updated_keys, updated_values

    def normalize(self, norm: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
        if not self.dynamic_rmsnorm:
            return norm(hidden)
        scale = hidden.abs().amax(dim=-1, keepdim=True).clamp_min(1.0)
        scaled = hidden / scale
        variance = (scaled * scaled).mean(dim=-1, keepdim=True) + norm.eps / scale / scale
        return (scaled / variance.sqrt()) * norm.alpha


class TemporalShard(nn.Module):
    """Consecutive real Temporal layers with separate per-layer K/V interfaces."""

    def __init__(self, blocks: list[TemporalBlock]) -> None:
        super().__init__()
        if not blocks:
            raise ValueError("A shard must contain at least one layer")
        self.blocks = nn.ModuleList(blocks)

    def forward(
        self, hidden: torch.Tensor, position: torch.Tensor, *caches: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        if len(caches) != 2 * len(self.blocks):
            raise ValueError("Expected one key and one value cache per layer")
        updated = []
        for index, block in enumerate(self.blocks):
            hidden, keys, values = block.forward_split(
                hidden, caches[2 * index], caches[2 * index + 1], position
            )
            updated.extend((keys, values))
        return hidden, *updated


class TemporalFrontend(nn.Module):
    """Inference-only sum of Moshi's text and audio-code embeddings."""

    def __init__(self, lm: nn.Module) -> None:
        super().__init__()
        self.audio_embeddings = lm.emb
        self.text_embedding = lm.text_emb
        self.audio_offset = lm.audio_offset

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        hidden = self.audio_embeddings[0](sequence[:, self.audio_offset])
        for index in range(1, len(self.audio_embeddings)):
            hidden = hidden + self.audio_embeddings[index](
                sequence[:, index + self.audio_offset]
            )
        return hidden + self.text_embedding(sequence[:, 0])


class TemporalHead(nn.Module):
    """Inference-only Temporal normalization and text projection."""

    def __init__(self, lm: nn.Module) -> None:
        super().__init__()
        self.out_norm = lm.out_norm
        self.text_linear = lm.text_linear

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.out_norm is not None:
            hidden = self.out_norm(hidden)
        return hidden, self.text_linear(hidden)[:, None]


class ExplicitTemporal(nn.Module):
    def __init__(self, lm: nn.Module) -> None:
        super().__init__()
        if lm.transformer.positional_embedding not in ("rope", "rope_concat", "none"):
            raise ValueError("Only rotary or no positional embeddings are supported")
        self.frontend = TemporalFrontend(lm)
        self.blocks = nn.ModuleList(TemporalBlock(layer) for layer in lm.transformer.layers)
        self.head = TemporalHead(lm)

    def embed(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.frontend(sequence)

    def forward(
        self, sequence: torch.Tensor, position: torch.Tensor, *caches: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        hidden = self.embed(sequence)
        updated = []
        for block, cache in zip(self.blocks, caches, strict=True):
            hidden, cache = block(hidden, cache, position)
            updated.append(cache)
        hidden, text_logits = self.head(hidden)
        return hidden, text_logits, position + 1, *updated


class ExplicitDepFormer(nn.Module):
    """Greedy, single-audio-frame DepFormer inference with no persistent state.

    DepFormer autoregressively predicts ``dep_q`` audio codebooks.  Its cache is
    local to one Temporal step in upstream ``LMGen.depformer_step`` and must be
    reset before the next audio frame.  Statically unrolling the short loop makes
    that lifetime explicit and removes Python control flow from the ONNX graph.
    """

    def __init__(self, lm: nn.Module) -> None:
        super().__init__()
        if lm.depformer is None or lm.depformer_text_emb is None:
            raise ValueError("The checkpoint has no DepFormer")
        if lm.depformer_emb is None:
            raise ValueError("The checkpoint has no DepFormer audio embeddings")
        if lm.dep_q < 1:
            raise ValueError("dep_q must be positive")
        if len(lm.depformer.layers) < 1:
            raise ValueError("The DepFormer must contain at least one layer")

        for layer in lm.depformer.layers:
            attention = layer.self_attn
            if layer.skip_self_attn or layer.cross_attention is not None:
                raise ValueError("Only causal DepFormer self-attention is supported")
            if not attention.causal or attention.context is None:
                raise ValueError("DepFormer requires a finite causal context")
            if attention.rope is not None:
                raise ValueError(
                    "This export expects the checkpoint's position-free DepFormer"
                )
            if attention.context < lm.dep_q:
                raise ValueError("DepFormer context is shorter than dep_q")
            if attention.weights_per_step != lm.dep_q:
                raise ValueError("Expected one DepFormer attention weight set per codebook")
            if layer.weights_per_step != lm.dep_q:
                raise ValueError("Expected one DepFormer feed-forward weight set per codebook")

        self.dep_q = lm.dep_q
        self.card = lm.card
        self.depformer_multi_linear = lm.depformer_multi_linear
        self.depformer_weights_per_step_schedule = lm.depformer_weights_per_step_schedule
        self.input_linears = lm.depformer_in
        self.text_embedding = lm.depformer_text_emb
        self.audio_embeddings = lm.depformer_emb
        self.layers = lm.depformer.layers
        self.output_norms = lm.depformer_norms
        self.output_linears = lm.linears

    @staticmethod
    def _empty_cache(
        layer: nn.Module, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attention = layer.self_attn
        heads = attention.num_heads // attention.kv_repeat
        head_dim = attention.embed_dim // attention.num_heads
        shape = (hidden.shape[0], heads, attention.context, head_dim)
        return hidden.new_zeros(shape), hidden.new_zeros(shape)

    @staticmethod
    def _block_step(
        layer: nn.Module,
        hidden: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        step: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attention = layer.self_attn
        projection_index = _weight_index(attention, step)
        projected = attention.in_projs[projection_index](layer.norm1(hidden))
        heads = attention.num_heads
        kv_heads = heads // attention.kv_repeat
        head_dim = attention.embed_dim // heads
        query_width = heads * head_dim
        kv_width = kv_heads * head_dim
        query, key, value = projected.split((query_width, kv_width, kv_width), dim=-1)
        query = query.reshape(hidden.shape[0], 1, heads, head_dim).transpose(1, 2)
        key = key.reshape(hidden.shape[0], 1, kv_heads, head_dim).transpose(1, 2)
        value = value.reshape(hidden.shape[0], 1, kv_heads, head_dim).transpose(1, 2)

        slots = torch.arange(attention.context, device=hidden.device)
        write_mask = (slots == step).reshape(1, 1, attention.context, 1)
        keys = torch.where(write_mask, key, key_cache)
        values = torch.where(write_mask, value, value_cache)
        attention_keys = keys
        attention_values = values
        if attention.kv_repeat > 1:
            attention_keys = attention_keys.repeat_interleave(attention.kv_repeat, dim=1)
            attention_values = attention_values.repeat_interleave(attention.kv_repeat, dim=1)
        valid = (slots <= step).reshape(1, 1, 1, attention.context)
        update = functional.scaled_dot_product_attention(
            query,
            attention_keys,
            attention_values,
            valid,
            dropout_p=0.0,
        )
        update = update.transpose(1, 2).reshape(hidden.shape[0], 1, query_width)
        update = attention.out_projs[projection_index](update)
        hidden = hidden.to(update) + layer.layer_scale_1(update)

        normalized = layer.norm2(hidden)
        if layer.gating is None:
            update = layer.linear2(layer.activation(layer.linear1(normalized)))
        else:
            gating_index = _weight_index(layer, step)
            update = layer.gating[gating_index](normalized)
        hidden = hidden.to(update) + layer.layer_scale_2(update)
        return hidden, keys, values

    def forward(
        self, text_token: torch.Tensor, temporal: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return greedy audio tokens and all pre-argmax audio logits.

        ``text_token`` is ``[B]`` and ``temporal`` is ``[B, 1, temporal_dim]``.
        Outputs are ``tokens [B, dep_q, 1]`` and
        ``logits [B, dep_q, 1, card]``.
        """
        caches = [self._empty_cache(layer, temporal) for layer in self.layers]
        previous = text_token
        tokens = []
        logits = []
        for step in range(self.dep_q):
            linear_index = step
            if self.depformer_weights_per_step_schedule is not None:
                linear_index = self.depformer_weights_per_step_schedule[step]
            if not self.depformer_multi_linear:
                linear_index = 0
            hidden = self.input_linears[linear_index](temporal)
            if step == 0:
                token_embedding = self.text_embedding(previous[:, None])
            else:
                token_embedding = self.audio_embeddings[step - 1](previous[:, None])
            hidden = hidden + token_embedding
            updated_caches = []
            for layer, (keys, values) in zip(self.layers, caches, strict=True):
                hidden, keys, values = self._block_step(
                    layer, hidden, keys, values, step
                )
                updated_caches.append((keys, values))
            caches = updated_caches
            step_logits = self.output_linears[step](self.output_norms[step](hidden))
            step_logits = step_logits[:, None]
            previous = step_logits.argmax(dim=-1)[:, 0, 0]
            tokens.append(previous[:, None, None])
            logits.append(step_logits)
        return torch.cat(tokens, dim=1), torch.cat(logits, dim=1)


class DepFormerStepBlock(nn.Module):
    """One real DepFormer layer specialized to one constant codebook index."""

    def __init__(self, layer: nn.Module, step: int) -> None:
        super().__init__()
        attention = layer.self_attn
        projection_index = _weight_index(attention, step)
        self.norm1 = layer.norm1
        self.in_proj = attention.in_projs[projection_index]
        self.out_proj = attention.out_projs[projection_index]
        self.norm2 = layer.norm2
        self.layer_scale_1 = layer.layer_scale_1
        self.layer_scale_2 = layer.layer_scale_2
        self.activation = layer.activation
        self.linear1 = layer.linear1
        self.linear2 = layer.linear2
        self.gating = (
            None if layer.gating is None else layer.gating[_weight_index(layer, step)]
        )
        self.step = step
        self.context = attention.context
        self.heads = attention.num_heads
        self.kv_heads = attention.num_heads // attention.kv_repeat
        self.kv_repeat = attention.kv_repeat
        self.head_dim = attention.embed_dim // attention.num_heads

    def empty_cache(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (hidden.shape[0], self.kv_heads, self.context, self.head_dim)
        return hidden.new_zeros(shape), hidden.new_zeros(shape)

    def forward(
        self,
        hidden: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        projected = self.in_proj(self.norm1(hidden))
        query_width = self.heads * self.head_dim
        kv_width = self.kv_heads * self.head_dim
        query, key, value = projected.split((query_width, kv_width, kv_width), dim=-1)
        query = query.reshape(
            hidden.shape[0], 1, self.heads, self.head_dim
        ).transpose(1, 2)
        key = key.reshape(
            hidden.shape[0], 1, self.kv_heads, self.head_dim
        ).transpose(1, 2)
        value = value.reshape(
            hidden.shape[0], 1, self.kv_heads, self.head_dim
        ).transpose(1, 2)
        slots = torch.arange(self.context, device=hidden.device)
        write_mask = (slots == self.step).reshape(1, 1, self.context, 1)
        keys = torch.where(write_mask, key, key_cache)
        values = torch.where(write_mask, value, value_cache)
        attention_keys = keys
        attention_values = values
        if self.kv_repeat > 1:
            attention_keys = attention_keys.repeat_interleave(self.kv_repeat, dim=1)
            attention_values = attention_values.repeat_interleave(
                self.kv_repeat, dim=1
            )
        valid = (slots <= self.step).reshape(1, 1, 1, self.context)
        update = functional.scaled_dot_product_attention(
            query, attention_keys, attention_values, valid, dropout_p=0.0
        )
        update = update.transpose(1, 2).reshape(
            hidden.shape[0], 1, query_width
        )
        update = self.out_proj(update)
        hidden = hidden.to(update) + self.layer_scale_1(update)
        normalized = self.norm2(hidden)
        if self.gating is None:
            update = self.linear2(self.activation(self.linear1(normalized)))
        else:
            update = self.gating(normalized)
        return hidden.to(update) + self.layer_scale_2(update), keys, values


class ExplicitDepFormerStep(nn.Module):
    """One codebook step with explicit per-layer DepFormer K/V tensors."""

    def __init__(self, depformer: ExplicitDepFormer, step: int) -> None:
        super().__init__()
        if not 0 <= step < depformer.dep_q:
            raise ValueError(f"DepFormer step must be within 0:{depformer.dep_q}")
        linear_index = step
        if depformer.depformer_weights_per_step_schedule is not None:
            linear_index = depformer.depformer_weights_per_step_schedule[step]
        if not depformer.depformer_multi_linear:
            linear_index = 0
        self.input_linear = depformer.input_linears[linear_index]
        self.token_embedding = (
            depformer.text_embedding
            if step == 0
            else depformer.audio_embeddings[step - 1]
        )
        self.blocks = nn.ModuleList(
            DepFormerStepBlock(layer, step) for layer in depformer.layers
        )
        self.output_norm = depformer.output_norms[step]
        self.output_linear = depformer.output_linears[step]

    def empty_caches(self, temporal: torch.Tensor) -> list[torch.Tensor]:
        return [
            value
            for block in self.blocks
            for value in block.empty_cache(temporal)
        ]

    def forward(
        self,
        previous_token: torch.Tensor,
        temporal: torch.Tensor,
        *caches: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        if len(caches) != 2 * len(self.blocks):
            raise ValueError("Expected one key and one value cache per DepFormer layer")
        hidden = self.input_linear(temporal) + self.token_embedding(
            previous_token[:, None]
        )
        updated = []
        for index, block in enumerate(self.blocks):
            hidden, keys, values = block(
                hidden, caches[2 * index], caches[2 * index + 1]
            )
            updated.extend((keys, values))
        logits = self.output_linear(self.output_norm(hidden))[:, None]
        token = logits.argmax(dim=-1)[:, 0, 0]
        return token, logits, *updated

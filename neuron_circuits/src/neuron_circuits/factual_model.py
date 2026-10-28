"""Small causal transformer with explicit post-SwiGLU intervention points."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class Intervention:
    layer: int
    kind: str = "mlp_channel"
    channel: int | None = None
    replacement: float = 0.0
    head: int | None = None
    direction: torch.Tensor | None = None


class Block(nn.Module):
    def __init__(self, width: int, heads: int, mlp_ratio: int = 2):
        super().__init__()
        self.width, self.heads, self.head_dim = width, heads, width // heads
        hidden = mlp_ratio * width
        self.attn_norm = nn.RMSNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.attn_out = nn.Linear(width, width, bias=False)
        self.mlp_norm = nn.RMSNorm(width)
        self.gate = nn.Linear(width, hidden, bias=False)
        self.up = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, x: torch.Tensor, intervention: Intervention | None = None, channel_mask: torch.Tensor | None = None):
        batch, length, width = x.shape
        normalized = self.attn_norm(x)
        query, key, value = self.qkv(normalized).chunk(3, dim=-1)
        def heads(tensor):
            return tensor.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        attended = F.scaled_dot_product_attention(heads(query), heads(key), heads(value), is_causal=True)
        attended = attended.transpose(1, 2)
        if intervention is not None and intervention.kind == "attention_head":
            attended = attended.clone()
            attended[:, -1, intervention.head] = intervention.replacement
        x = x + self.attn_out(attended.reshape(batch, length, width))
        normalized = self.mlp_norm(x)
        activation = F.silu(self.gate(normalized)) * self.up(normalized)
        if channel_mask is not None:
            activation = activation * channel_mask.to(activation)[None, None, :]
        if intervention is not None and intervention.kind == "mlp_channel":
            activation = activation.clone()
            if intervention.channel is None:
                activation[:, -1, :] = intervention.replacement
            else:
                activation[:, -1, intervention.channel] = intervention.replacement
        elif intervention is not None and intervention.kind == "mlp_direction":
            direction = intervention.direction.to(activation)
            activation = activation.clone()
            current = activation[:, -1] @ direction
            activation[:, -1] += (intervention.replacement - current)[:, None] * direction[None, :]
        return x + self.down(activation), activation, attended


class FactualTransformer(nn.Module):
    def __init__(self, vocab_size: int, width: int, heads: int, layers: int = 4, max_length: int = 24):
        super().__init__()
        self.width, self.heads, self.layers_count = width, heads, layers
        self.embedding = nn.Embedding(vocab_size, width)
        self.position = nn.Parameter(torch.empty(max_length, width))
        self.blocks = nn.ModuleList([Block(width, heads) for _ in range(layers)])
        self.final_norm = nn.RMSNorm(width)
        self.answer_head = nn.Linear(width, 8, bias=False)
        self.reset_parameters()

    @property
    def channels(self) -> int:
        return 2 * self.width

    def reset_parameters(self) -> None:
        nn.init.normal_(self.position, std=0.02)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=0.02)
        for block in self.blocks:
            nn.init.normal_(block.attn_out.weight, std=0.02 / math.sqrt(2 * self.layers_count))
            nn.init.normal_(block.down.weight, std=0.02 / math.sqrt(2 * self.layers_count))

    def forward(self, tokens: torch.Tensor, *, intervention: Intervention | None = None, channel_masks: list[torch.Tensor] | None = None, return_activations: bool = False):
        hidden = self.embedding(tokens) + self.position[:tokens.shape[1]]
        mlp_activations, attention_outputs = [], []
        for layer, block in enumerate(self.blocks):
            active = intervention if intervention is not None and intervention.layer == layer else None
            mask = None if channel_masks is None else channel_masks[layer]
            hidden, mlp, attention = block(hidden, active, mask)
            if return_activations:
                mlp_activations.append(mlp[:, -1].float())
                attention_outputs.append(attention[:, -1].float())
        logits = self.answer_head(self.final_norm(hidden[:, -1])).float()
        if return_activations:
            return logits, mlp_activations, attention_outputs
        return logits

    def component_counts(self) -> dict[str, int]:
        groups = {
            "token_embedding": self.embedding.num_embeddings * self.embedding.embedding_dim,
            "position_embedding": self.position.numel(), "attention": 0, "mlp": 0,
            "norms": self.final_norm.weight.numel(), "answer_head": self.answer_head.weight.numel(),
        }
        for block in self.blocks:
            groups["attention"] += sum(parameter.numel() for module in (block.qkv, block.attn_out) for parameter in module.parameters())
            groups["mlp"] += sum(parameter.numel() for module in (block.gate, block.up, block.down) for parameter in module.parameters())
            groups["norms"] += block.attn_norm.weight.numel() + block.mlp_norm.weight.numel()
        groups["total"] = sum(parameter.numel() for parameter in self.parameters())
        return groups

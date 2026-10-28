"""Small decoder transformer for the compositional mini-language."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class ActivationPatch:
    layer: int
    values: torch.Tensor
    channels: torch.Tensor | None = None
    additive: bool = False


class GrammarBlock(nn.Module):
    def __init__(self, width: int, heads: int, mlp_ratio: int, mlp_kind: str):
        super().__init__()
        if mlp_kind not in {"swiglu", "bilinear"}:
            raise ValueError(mlp_kind)
        self.width, self.heads, self.head_dim = width, heads, width // heads
        self.mlp_kind = mlp_kind
        hidden = mlp_ratio * width
        self.attn_norm = nn.RMSNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.attn_out = nn.Linear(width, width, bias=False)
        self.mlp_norm = nn.RMSNorm(width)
        self.left = nn.Linear(width, hidden, bias=False)
        self.right = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        patch: ActivationPatch | None = None,
        channel_mask: torch.Tensor | None = None,
        channel_projector: torch.Tensor | None = None,
    ):
        batch, length, width = x.shape
        normalized = self.attn_norm(x)
        query, key, value = self.qkv(normalized).chunk(3, dim=-1)

        def heads(tensor):
            return tensor.view(batch, length, self.heads, self.head_dim).transpose(1, 2)

        attended = F.scaled_dot_product_attention(heads(query), heads(key), heads(value), is_causal=True)
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        x = x + self.attn_out(attended)
        normalized = self.mlp_norm(x)
        left, right = self.left(normalized), self.right(normalized)
        activation = F.silu(left) * right if self.mlp_kind == "swiglu" else left * right
        if patch is not None:
            activation = activation.clone()
            current = activation[:, -1]
            values = patch.values.to(current)
            if values.ndim == 1:
                values = values[None, :].expand_as(current)
            if patch.channels is None:
                activation[:, -1] = current + values if patch.additive else values
            else:
                channels = patch.channels.to(device=current.device, dtype=torch.bool)
                if patch.additive:
                    activation[:, -1, channels] += values[:, channels]
                else:
                    activation[:, -1, channels] = values[:, channels]
        if channel_mask is not None:
            mask = channel_mask.to(activation)
            if mask.ndim == 1:
                mask = mask[None, None, :]
            elif mask.ndim == 2:
                if mask.shape[0] != activation.shape[0]:
                    raise ValueError("batched channel mask must match activation batch")
                mask = mask[:, None, :]
            else:
                raise ValueError("channel mask must have shape [channels] or [batch, channels]")
            activation = activation * mask
        if channel_projector is not None:
            activation = activation @ channel_projector.to(activation)
        return x + self.down(activation), activation


class GrammarTransformer(nn.Module):
    def __init__(
        self, vocab_size: int, width: int = 96, heads: int = 3, layers: int = 2,
        mlp_ratio: int = 2, mlp_kind: str = "swiglu", max_length: int = 16,
    ):
        super().__init__()
        self.width, self.heads, self.layers_count = width, heads, layers
        self.mlp_ratio, self.mlp_kind = mlp_ratio, mlp_kind
        self.embedding = nn.Embedding(vocab_size, width)
        self.position = nn.Parameter(torch.empty(max_length, width))
        self.blocks = nn.ModuleList([
            GrammarBlock(width, heads, mlp_ratio, mlp_kind) for _ in range(layers)
        ])
        self.final_norm = nn.RMSNorm(width)
        self.unembedding = nn.Linear(width, vocab_size, bias=False)
        self.reset_parameters()

    @property
    def channels(self) -> int:
        return self.mlp_ratio * self.width

    def reset_parameters(self):
        nn.init.normal_(self.position, std=.02)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=.02)
        for block in self.blocks:
            nn.init.normal_(block.attn_out.weight, std=.02 / math.sqrt(2 * self.layers_count))
            nn.init.normal_(block.down.weight, std=.02 / math.sqrt(2 * self.layers_count))

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        patch: ActivationPatch | None = None,
        return_activations: bool = False,
        return_full_activations: bool = False,
        channel_masks: dict[int, torch.Tensor] | None = None,
        channel_projectors: dict[int, torch.Tensor] | None = None,
    ):
        hidden = self.embedding(tokens) + self.position[:tokens.shape[1]]
        activations = []
        for layer, block in enumerate(self.blocks):
            active = patch if patch is not None and patch.layer == layer else None
            mask = channel_masks.get(layer) if channel_masks is not None else None
            projector = channel_projectors.get(layer) if channel_projectors is not None else None
            if mask is not None and projector is not None:
                raise ValueError(f"layer {layer} has both a channel mask and projector")
            hidden, activation = block(hidden, active, mask, projector)
            if return_full_activations:
                activations.append(activation.float())
            elif return_activations:
                activations.append(activation[:, -1].float())
        logits = self.unembedding(self.final_norm(hidden)).float()
        return (logits, activations) if return_activations or return_full_activations else logits

    def component_counts(self) -> dict[str, int]:
        output = {"embedding": self.embedding.weight.numel() + self.position.numel(),
                  "attention": 0, "mlp": 0,
                  "norms": self.final_norm.weight.numel(), "unembedding": self.unembedding.weight.numel()}
        for block in self.blocks:
            output["attention"] += block.qkv.weight.numel() + block.attn_out.weight.numel()
            output["mlp"] += block.left.weight.numel() + block.right.weight.numel() + block.down.weight.numel()
            output["norms"] += block.attn_norm.weight.numel() + block.mlp_norm.weight.numel()
        output["total"] = sum(parameter.numel() for parameter in self.parameters())
        return output

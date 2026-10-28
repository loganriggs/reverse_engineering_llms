"""Fresh-forward counterfactual patching for Qwen2-family attention internals."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Literal

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from transformers.models.qwen2 import modeling_qwen2


AttentionComponent = Literal[
    "query", "key", "value", "pattern", "message", "head_output"
]


@dataclass(frozen=True)
class AttentionPatch:
    component: AttentionComponent
    mask: Tensor  # [layer, position, head], True means copy donor
    donor: tuple[Tensor, ...]


@dataclass
class AttentionTrace:
    logits: Tensor
    query: tuple[Tensor, ...]       # each [batch, position, query_head, head_dim]
    key: tuple[Tensor, ...]         # each [batch, position, kv_head, head_dim]
    value: tuple[Tensor, ...]       # each [batch, position, kv_head, head_dim]
    pattern: tuple[Tensor, ...]     # each [batch, query_head, destination, source]
    message: tuple[Tensor, ...]     # each [batch, query_head, destination, source, head_dim]
    head_output: tuple[Tensor, ...] # each [batch, position, query_head, head_dim]

    def component(self, name: AttentionComponent) -> tuple[Tensor, ...]:
        return getattr(self, name)


class QwenAttentionPatchAdapter:
    """Patch attention activations while recomputing all downstream modules."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        layers = getattr(getattr(model, "model", None), "layers", None)
        if layers is None:
            raise TypeError("expected a Qwen2-family model with model.layers")
        self.attentions = tuple(layer.self_attn for layer in layers)
        if not self.attentions or not all(
            all(hasattr(attn, name) for name in ("q_proj", "k_proj", "v_proj", "o_proj"))
            for attn in self.attentions
        ):
            raise TypeError("attention modules do not expose q/k/v/o projections")
        self.num_layers = len(self.attentions)
        self.num_query_heads = self.attentions[0].config.num_attention_heads
        self.num_kv_heads = self.attentions[0].config.num_key_value_heads
        self.head_dim = self.attentions[0].head_dim
        for attention in self.attentions:
            attention.config._attn_implementation = "eager"
        self.model.config._attn_implementation = "eager"

    def unit_shape(self, component: AttentionComponent, positions: int) -> tuple[int, ...]:
        if component == "message":
            return self.num_layers, self.num_query_heads, positions, positions
        heads = self.num_kv_heads if component in ("key", "value") else self.num_query_heads
        return self.num_layers, positions, heads

    @staticmethod
    def _patch_vectors(
        tensor: Tensor,
        layer: int,
        heads: int,
        patch: AttentionPatch | None,
    ) -> Tensor:
        if patch is None:
            return tensor
        donor = patch.donor[layer].to(device=tensor.device, dtype=tensor.dtype)
        selected = patch.mask[layer].to(device=tensor.device, dtype=torch.bool)
        view = tensor.view(*tensor.shape[:-1], heads, -1)
        return torch.where(selected[None, :, :, None], donor, view).reshape_as(tensor)

    @contextmanager
    def _hooks(
        self,
        patch: AttentionPatch | tuple[AttentionPatch, ...] | None,
        retain_grads: bool,
    ) -> Iterator[dict[str, list[Tensor | None]]]:
        patches = () if patch is None else (patch if isinstance(patch, tuple) else (patch,))
        patch_by_component = {item.component: item for item in patches}
        if len(patch_by_component) != len(patches):
            raise ValueError("at most one patch per attention component is supported")
        captures: dict[str, list[Tensor | None]] = {
            name: [None] * self.num_layers
            for name in ("query", "key", "value", "pattern", "message", "head_output")
        }
        handles = []

        def projection_hook(name: AttentionComponent, layer: int, heads: int):
            def hook(_module: nn.Module, _args: tuple[Tensor, ...], output: Tensor):
                used = output
                component_patch = patch_by_component.get(name)
                if component_patch is not None:
                    used = self._patch_vectors(output, layer, heads, component_patch)
                shaped = used.view(*used.shape[:-1], heads, self.head_dim)
                if retain_grads:
                    shaped.retain_grad()
                captures[name][layer] = shaped
                # Return through the shaped view so it is part of the graph used
                # downstream and gradients with respect to the head basis exist.
                return shaped.reshape_as(output)
            return hook

        def output_pre_hook(layer: int):
            def hook(_module: nn.Module, args: tuple[Tensor, ...]):
                original = args[0]
                used = original
                component_patch = patch_by_component.get("head_output")
                if component_patch is not None:
                    used = self._patch_vectors(
                        original, layer, self.num_query_heads, component_patch
                    )
                shaped = used.view(
                    *used.shape[:-1], self.num_query_heads, self.head_dim
                )
                if retain_grads:
                    shaped.retain_grad()
                captures["head_output"][layer] = shaped
                return (shaped.reshape_as(original), *args[1:])
            return hook

        original_eager = modeling_qwen2.eager_attention_forward

        def patched_eager(
            module: nn.Module,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            attention_mask: Tensor | None,
            scaling: float,
            dropout: float = 0.0,
            **_kwargs,
        ):
            key_states = modeling_qwen2.repeat_kv(key, module.num_key_value_groups)
            value_states = modeling_qwen2.repeat_kv(value, module.num_key_value_groups)
            weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
            if attention_mask is not None:
                weights = weights + attention_mask[:, :, :, : key_states.shape[-2]]
            weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
            weights = F.dropout(weights, p=dropout, training=module.training)
            layer = module.layer_idx
            used = weights
            component_patch = patch_by_component.get("pattern")
            if component_patch is not None:
                donor = component_patch.donor[layer].to(device=weights.device, dtype=weights.dtype)
                selected = component_patch.mask[layer].to(device=weights.device, dtype=torch.bool)
                used = torch.where(selected.T[None, :, :, None], donor, weights)
            if retain_grads:
                used.retain_grad()
            captures["pattern"][layer] = used
            messages = used[..., None] * value_states[:, :, None, :, :]
            captures["message"][layer] = messages
            output = torch.matmul(used, value_states)
            message_patch = patch_by_component.get("message")
            if message_patch is not None:
                donor = message_patch.donor[layer].to(
                    device=messages.device, dtype=messages.dtype
                )
                selected = message_patch.mask[layer].to(
                    device=messages.device, dtype=torch.bool
                )
                delta = torch.where(
                    selected[None, :, :, :, None], donor - messages, 0.0
                )
                output = output + delta.sum(dim=3)
            return output.transpose(1, 2).contiguous(), used

        try:
            for layer, attention in enumerate(self.attentions):
                handles.extend((
                    attention.q_proj.register_forward_hook(
                        projection_hook("query", layer, self.num_query_heads)
                    ),
                    attention.k_proj.register_forward_hook(
                        projection_hook("key", layer, self.num_kv_heads)
                    ),
                    attention.v_proj.register_forward_hook(
                        projection_hook("value", layer, self.num_kv_heads)
                    ),
                    attention.o_proj.register_forward_pre_hook(output_pre_hook(layer)),
                ))
            modeling_qwen2.eager_attention_forward = patched_eager
            yield captures
        finally:
            modeling_qwen2.eager_attention_forward = original_eager
            for handle in handles:
                handle.remove()

    def forward(
        self,
        input_ids: Tensor,
        patch: AttentionPatch | tuple[AttentionPatch, ...] | None = None,
        *,
        retain_grads: bool = False,
    ) -> AttentionTrace:
        with self._hooks(patch, retain_grads) as captures:
            output = self.model(input_ids=input_ids, use_cache=False)
        if any(item is None for values in captures.values() for item in values):
            raise RuntimeError("failed to capture every attention activation")
        return AttentionTrace(
            logits=output.logits[:, -1, :],
            **{name: tuple(values) for name, values in captures.items()},
        )


def counterfactual_scores(
    base: AttentionTrace,
    donor: AttentionTrace,
    target_token_id: int,
    original_token_id: int,
) -> dict[AttentionComponent, Tensor]:
    """Signed first-order score for copying each donor attention unit."""
    components: tuple[AttentionComponent, ...] = (
        "query", "key", "value", "pattern", "head_output"
    )
    tensors = [tensor for name in components for tensor in base.component(name)]
    objective = base.logits[0, target_token_id] - base.logits[0, original_token_id]
    gradients = torch.autograd.grad(objective, tensors)
    output = {}
    cursor = 0
    for name in components:
        layer_scores = []
        for base_tensor, donor_tensor in zip(
            base.component(name), donor.component(name), strict=True
        ):
            gradient = gradients[cursor]
            cursor += 1
            difference = donor_tensor.detach() - base_tensor.detach()
            contribution = gradient.detach().float() * difference.float()
            if name == "pattern":
                score = contribution.sum(dim=-1)[0].transpose(0, 1)
            else:
                score = contribution.sum(dim=-1)[0]
            layer_scores.append(score)
        output[name] = torch.stack(layer_scores)
    # A message is A[t,s] * V[s]. Its first-order effect uses the gradient
    # of the summed per-head output; no expanded message tensor needs to be
    # inserted into the intact graph, preserving the model's matmul numerics.
    message_scores = []
    head_output_start = sum(len(base.component(name)) for name in components[:-1])
    for layer, (base_message, donor_message) in enumerate(
        zip(base.message, donor.message, strict=True)
    ):
        head_gradient = gradients[head_output_start + layer][0].transpose(0, 1)
        difference = donor_message.detach()[0] - base_message.detach()[0]
        message_scores.append(
            (difference.float() * head_gradient.float()[:, :, None, :]).sum(dim=-1)
        )
    output["message"] = torch.stack(message_scores)
    return output


def mask_from_unit_ranking(
    ranking: Tensor,
    patched: int,
    shape: tuple[int, ...],
) -> Tensor:
    mask = torch.zeros(int(torch.tensor(shape).prod()), dtype=torch.bool, device=ranking.device)
    if patched:
        mask[ranking[:patched]] = True
    return mask.view(shape)

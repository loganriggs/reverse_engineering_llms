"""Counterfactual patching of whole attention and MLP module outputs."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Literal

import torch
from torch import Tensor, nn


ModuleComponent = Literal["attention_output", "mlp_output"]


@dataclass(frozen=True)
class ModulePatch:
    component: ModuleComponent
    mask: Tensor  # [layer, position], True means copy donor
    donor: tuple[Tensor, ...]


@dataclass
class ModuleTrace:
    logits: Tensor
    attention_output: tuple[Tensor, ...]
    mlp_output: tuple[Tensor, ...]

    def component(self, name: ModuleComponent) -> tuple[Tensor, ...]:
        return getattr(self, name)


class QwenModulePatchAdapter:
    """Patch projected module writes while recomputing every downstream layer."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        layers = getattr(getattr(model, "model", None), "layers", None)
        if layers is None:
            raise TypeError("expected a Qwen2-family model with model.layers")
        self.layers = tuple(layers)
        self.attention_outputs = tuple(layer.self_attn.o_proj for layer in self.layers)
        self.mlp_outputs = tuple(layer.mlp.down_proj for layer in self.layers)
        self.num_layers = len(self.layers)

    def unit_shape(self, positions: int) -> tuple[int, int]:
        return self.num_layers, positions

    @contextmanager
    def _hooks(
        self,
        patches: ModulePatch | tuple[ModulePatch, ...] | None,
        retain_grads: bool,
    ) -> Iterator[dict[str, list[Tensor | None]]]:
        patch_items = () if patches is None else (patches if isinstance(patches, tuple) else (patches,))
        by_component = {patch.component: patch for patch in patch_items}
        if len(by_component) != len(patch_items):
            raise ValueError("at most one patch per module component is supported")
        captures = {
            "attention_output": [None] * self.num_layers,
            "mlp_output": [None] * self.num_layers,
        }
        handles = []

        def hook(component: ModuleComponent, layer: int):
            def apply(_module: nn.Module, _args: tuple[Tensor, ...], output: Tensor):
                used = output
                patch = by_component.get(component)
                if patch is not None:
                    donor = patch.donor[layer].to(device=output.device, dtype=output.dtype)
                    selected = patch.mask[layer].to(device=output.device, dtype=torch.bool)
                    used = torch.where(selected[None, :, None], donor, output)
                if retain_grads:
                    used.retain_grad()
                captures[component][layer] = used
                return used
            return apply

        try:
            for layer, (attention, mlp) in enumerate(zip(self.attention_outputs, self.mlp_outputs)):
                handles.append(attention.register_forward_hook(hook("attention_output", layer)))
                handles.append(mlp.register_forward_hook(hook("mlp_output", layer)))
            yield captures
        finally:
            for handle in handles:
                handle.remove()

    def forward(
        self,
        input_ids: Tensor,
        patches: ModulePatch | tuple[ModulePatch, ...] | None = None,
        *,
        retain_grads: bool = False,
    ) -> ModuleTrace:
        with self._hooks(patches, retain_grads) as captures:
            output = self.model(input_ids=input_ids, use_cache=False)
        if any(item is None for values in captures.values() for item in values):
            raise RuntimeError("failed to capture every module output")
        return ModuleTrace(
            logits=output.logits[:, -1, :],
            attention_output=tuple(captures["attention_output"]),
            mlp_output=tuple(captures["mlp_output"]),
        )


def module_counterfactual_scores(
    base: ModuleTrace,
    donor: ModuleTrace,
    target_token_id: int,
    original_token_id: int,
) -> dict[ModuleComponent, Tensor]:
    components: tuple[ModuleComponent, ...] = ("attention_output", "mlp_output")
    tensors = [tensor for component in components for tensor in base.component(component)]
    objective = base.logits[0, target_token_id] - base.logits[0, original_token_id]
    gradients = torch.autograd.grad(objective, tensors)
    output = {}
    cursor = 0
    for component in components:
        scores = []
        for base_tensor, donor_tensor in zip(base.component(component), donor.component(component)):
            gradient = gradients[cursor]
            cursor += 1
            scores.append(
                (gradient.detach().float() * (donor_tensor.detach().float() - base_tensor.detach().float())).sum(dim=-1)[0]
            )
        output[component] = torch.stack(scores)
    return output


def module_mask(shape: tuple[int, int], selected_flat: Tensor) -> Tensor:
    mask = torch.zeros(shape[0] * shape[1], dtype=torch.bool, device=selected_flat.device)
    if selected_flat.numel():
        mask[selected_flat] = True
    return mask.view(shape)

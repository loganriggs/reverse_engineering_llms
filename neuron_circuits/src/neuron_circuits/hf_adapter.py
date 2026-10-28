"""Fresh-forward Stage A masking for Hugging Face gated-MLP transformers."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch
from torch import Tensor, nn


@dataclass
class HFForward:
    logits: Tensor
    activations: tuple[Tensor, ...]


def _transformer_layers(model: nn.Module) -> list[nn.Module]:
    candidates = (
        ("model", "layers"),
        ("gpt_neox", "layers"),
        ("transformer", "h"),
    )
    for parent_name, layers_name in candidates:
        parent = getattr(model, parent_name, None)
        layers = getattr(parent, layers_name, None) if parent is not None else None
        if layers is not None:
            return list(layers)
    raise TypeError("unsupported transformer layout: cannot locate decoder layers")


def _mlp_output_projection(layer: nn.Module) -> nn.Linear:
    mlp = getattr(layer, "mlp", None)
    if mlp is None:
        raise TypeError("decoder layer has no MLP module")
    for name in ("down_proj", "dense_4h_to_h", "c_proj"):
        projection = getattr(mlp, name, None)
        if isinstance(projection, nn.Linear):
            return projection
    raise TypeError("unsupported MLP: cannot locate output projection")


class GatedMLPMaskAdapter:
    """Masks each MLP output-projection input without caching downstream state."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.layers = _transformer_layers(model)
        self.output_projections = tuple(_mlp_output_projection(layer) for layer in self.layers)
        widths = {projection.in_features for projection in self.output_projections}
        if len(widths) != 1:
            raise ValueError("all MLP layers must use one intermediate width")
        self.neuron_width = widths.pop()

    def mask_shape(self, positions: int) -> tuple[int, int, int]:
        return len(self.layers), positions, self.neuron_width

    @contextmanager
    def _hooks(
        self,
        mask: Tensor | None,
        baseline: Tensor | None,
        retain_activation_grads: bool,
    ) -> Iterator[list[Tensor]]:
        activations: list[Tensor] = []
        handles = []

        def make_hook(layer_index: int):
            def hook(_module: nn.Module, args: tuple[Tensor, ...]):
                hidden = args[0]
                if retain_activation_grads:
                    hidden.retain_grad()
                activations.append(hidden)
                if mask is None:
                    return None
                selected = mask[layer_index]
                if selected.ndim == 2:
                    selected = selected.unsqueeze(0)
                selected = selected.to(device=hidden.device, dtype=hidden.dtype)
                replacement: Tensor | float
                replacement = (
                    0.0
                    if baseline is None
                    else baseline[layer_index].to(device=hidden.device, dtype=hidden.dtype)
                )
                return (selected * hidden + (1.0 - selected) * replacement, *args[1:])
            return hook

        try:
            for index, projection in enumerate(self.output_projections):
                handles.append(projection.register_forward_pre_hook(make_hook(index)))
            yield activations
        finally:
            for handle in handles:
                handle.remove()

    def forward(
        self,
        input_ids: Tensor,
        mask: Tensor | None = None,
        *,
        baseline: Tensor | None = None,
        retain_activation_grads: bool = False,
    ) -> HFForward:
        """Execute a complete forward; intact caches are never accepted or reused."""
        with self._hooks(mask, baseline, retain_activation_grads) as activations:
            output = self.model(input_ids=input_ids, use_cache=False)
        return HFForward(output.logits[:, -1, :], tuple(activations))

    def write_norms(self) -> Tensor:
        return torch.stack(
            [projection.weight.float().norm(dim=0) for projection in self.output_projections]
        )


def validate_all_one_mask(
    adapter: GatedMLPMaskAdapter,
    input_ids: Tensor,
    *,
    atol: float = 2e-3,
    rtol: float = 2e-3,
) -> float:
    """Return max logit error and raise if an all-one mask changes the model."""
    with torch.no_grad():
        intact = adapter.forward(input_ids).logits.float()
        shape = adapter.mask_shape(input_ids.shape[1])
        all_one = torch.ones(shape, device=input_ids.device)
        masked = adapter.forward(input_ids, all_one).logits.float()
    error = float((intact - masked).abs().max())
    if not torch.allclose(intact, masked, atol=atol, rtol=rtol):
        raise AssertionError(f"all-one mask changed logits (max error {error:.6g})")
    return error


def validate_downstream_recomputation(
    adapter: GatedMLPMaskAdapter,
    input_ids: Tensor,
    *,
    layer: int = 0,
) -> tuple[tuple[int, int, int], float]:
    """Ablate one active upstream unit and verify later activations change."""
    if layer >= len(adapter.layers) - 1:
        raise ValueError("probe layer must precede at least one downstream layer")
    with torch.no_grad():
        intact = adapter.forward(input_ids)
        source = intact.activations[layer][0].abs()
        position, channel = (int(index) for index in torch.unravel_index(source.argmax(), source.shape))
        shape = adapter.mask_shape(input_ids.shape[1])
        mask = torch.ones(shape, device=input_ids.device)
        mask[layer, position, channel] = 0
        ablated = adapter.forward(input_ids, mask)
        change = float(
            (intact.activations[-1].float() - ablated.activations[-1].float()).abs().max()
        )
    if change == 0.0:
        raise AssertionError("upstream ablation did not change downstream MLP activations")
    return (layer, position, channel), change

"""Location restrictions for position-specific MLP counterfactual patches."""

from __future__ import annotations

import torch


CONDITIONS = (
    "exclude_final_layer",
    "exclude_final_four_layers",
    "exclude_final_position",
    "exclude_final_four_layers_and_position",
)


def eligible_instance_mask(
    shape: tuple[int, int, int], condition: str, device: str
) -> torch.Tensor:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown restriction: {condition}")
    layers, positions, _width = shape
    allowed = torch.ones(shape, dtype=torch.bool, device=device)
    if condition == "exclude_final_layer":
        allowed[layers - 1] = False
    if condition in (
        "exclude_final_four_layers",
        "exclude_final_four_layers_and_position",
    ):
        allowed[max(0, layers - 4):] = False
    if condition in (
        "exclude_final_position",
        "exclude_final_four_layers_and_position",
    ):
        allowed[:, positions - 1] = False
    return allowed


def restricted_ranking(scores: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    if scores.shape != allowed.shape:
        raise ValueError("scores and eligibility mask must have the same shape")
    eligible = torch.nonzero(allowed.reshape(-1), as_tuple=False).reshape(-1)
    order = torch.argsort(scores.reshape(-1)[eligible], descending=True, stable=True)
    return eligible[order]

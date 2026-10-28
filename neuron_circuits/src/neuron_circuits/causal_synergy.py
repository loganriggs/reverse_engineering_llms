"""Utilities for exact pairwise causal interaction and discrete information synergy."""

from __future__ import annotations

import numpy as np
import torch


def factorial_statistics(margins: torch.Tensor, material_threshold: float = .05) -> dict[str, torch.Tensor]:
    """Summarize `[m11,m10,m01,m00]` margin grids along the last axis."""
    if margins.shape[-1] != 4:
        raise ValueError("factorial margins must end in four states: 11,10,01,00")
    m11, m10, m01, m00 = margins.unbind(-1)
    interaction = m11 - m10 - m01 + m00
    a_alone, b_alone = m10 - m00, m01 - m00
    a_with_b, b_with_a = m11 - m01, m11 - m10
    denominator = a_alone.abs() + b_alone.abs() + interaction.abs() + 1e-8
    a_flip = ((a_alone * a_with_b < 0) &
              (torch.minimum(a_alone.abs(), a_with_b.abs()) > material_threshold))
    b_flip = ((b_alone * b_with_a < 0) &
              (torch.minimum(b_alone.abs(), b_with_a.abs()) > material_threshold))
    return {
        "interaction": interaction,
        "interaction_fraction": interaction.abs() / denominator,
        "a_alone": a_alone, "b_alone": b_alone,
        "a_with_b": a_with_b, "b_with_a": b_with_a,
        "sign_flip": a_flip | b_flip,
    }


def discrete_mutual_information(target: np.ndarray, code: np.ndarray) -> float:
    """Plug-in mutual information in bits for integer target and 1-D integer code."""
    target = np.asarray(target); code = np.asarray(code)
    if target.ndim != 1 or code.ndim != 1 or len(target) != len(code):
        raise ValueError("target and code must be equal-length vectors")
    _, y = np.unique(target, return_inverse=True)
    _, x = np.unique(code, return_inverse=True)
    joint = np.zeros((x.max() + 1, y.max() + 1), dtype=np.float64)
    np.add.at(joint, (x, y), 1)
    joint /= len(x)
    px, py = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    expected = px @ py
    keep = joint > 0
    return float(np.sum(joint[keep] * np.log2(joint[keep] / expected[keep])))


def conditional_mutual_information(
    target: np.ndarray, code: np.ndarray, context: np.ndarray | None = None,
) -> float:
    """Weighted plug-in `I(target; code | context)` in bits."""
    target, code = np.asarray(target), np.asarray(code)
    if context is None:
        return discrete_mutual_information(target, code)
    context = np.asarray(context)
    output = 0.0
    for value in np.unique(context):
        keep = context == value
        output += keep.mean() * discrete_mutual_information(target[keep], code[keep])
    return float(output)


def pair_information_synergy(
    target: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    context: np.ndarray | None = None,
) -> float:
    """Whole-minus-parts information synergy for two binary codes."""
    first = np.asarray(first, dtype=np.int64)
    second = np.asarray(second, dtype=np.int64)
    joint = first + 2 * second
    return (
        conditional_mutual_information(target, joint, context)
        - conditional_mutual_information(target, first, context)
        - conditional_mutual_information(target, second, context)
    )


def pair_complementary_information(
    target: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    context: np.ndarray | None = None,
) -> float:
    """Extra joint information beyond the better individual binary code."""
    first = np.asarray(first, dtype=np.int64)
    second = np.asarray(second, dtype=np.int64)
    joint = first + 2 * second
    individual = max(
        conditional_mutual_information(target, first, context),
        conditional_mutual_information(target, second, context),
    )
    return conditional_mutual_information(target, joint, context) - individual

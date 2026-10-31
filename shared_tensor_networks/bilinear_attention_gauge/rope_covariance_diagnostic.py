"""Weight-only RoPE commutator diagnostic for the saved rank-96 head cuts."""

from __future__ import annotations

import json

import numpy as np

from flagship_full_model_eval import get_cuts
from flagship_matrix_free import HD, rotate, rotary
from flagship_rank96_group_stability import CUTS
from flagship_transfer import ART


def run():
    results = []
    cuts = get_cuts()
    for cut_name in CUTS:
        x = cuts[cut_name]
        left, values, right = np.linalg.svd(x, full_matrices=False)
        a = left[:, :96] * np.sqrt(values[:96])
        b = right[:96].T * np.sqrt(values[:96])
        gauges = np.load(ART / f"head0_exact_gauge_rotations_{cut_name}.npz")
        for shift in (1, 8, 24):
            cos, sin = rotary(shift + 1)
            row_rotation = rotate(np.eye(HD), np.full(HD, shift), cos, sin)
            for label, selector in (
                ("full_cut", np.eye(96)),
                ("sparse_axis_0", gauges["intervention_activation_0"][:, :1]),
                ("random_axis_0", gauges["random_0"][:, :1]),
                ("sparse_axes_0_to_3", gauges["intervention_activation_0"][:, :4]),
            ):
                component = (a @ selector) @ (b @ selector).T
                transported = row_rotation.T @ component @ row_rotation
                results.append(dict(
                    cut=cut_name, shift=shift, component=label,
                    relative_frobenius_change=float(
                        np.linalg.norm(transported - component) /
                        np.linalg.norm(component)),
                ))
    result = dict(
        definition="Frobenius norm of R.T @ X @ R - X divided by norm X; R acts on row activations",
        notes="full_cut is rank 96; selected axes are examples, not an optimized group comparison",
        results=results,
    )
    (ART / "head0_rope_covariance_diagnostic.json").write_text(
        json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    run()

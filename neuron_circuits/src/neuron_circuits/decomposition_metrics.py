"""Metrics for comparing compressed or interpretable layer replacements."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


def bootstrap_mean_ci(values, *, seed=0, resamples=2000, confidence=0.95):
    """Return a row-bootstrap mean and percentile confidence interval."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array):
        raise ValueError("values must be a nonempty vector")
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, len(array), size=(resamples, len(array)))
    means = array[draws].mean(axis=1)
    tail = (1 - confidence) / 2
    return {
        "mean": float(array.mean()),
        "ci_low": float(np.quantile(means, tail)),
        "ci_high": float(np.quantile(means, 1 - tail)),
        "rows": int(len(array)),
    }


def effect_agreement(teacher_base, teacher_changed, student_base, student_changed):
    """Compare finite effects after removing the softmax-constant direction.

    Inputs have shape ``[..., vocabulary]`` and may be logits or log
    probabilities. Centering each effect makes the score invariant to a
    vocabulary-wide additive shift.
    """
    teacher = (teacher_changed - teacher_base).float()
    student = (student_changed - student_base).float()
    teacher = teacher - teacher.mean(dim=-1, keepdim=True)
    student = student - student.mean(dim=-1, keepdim=True)
    error = (student - teacher).square().sum()
    teacher_energy = teacher.square().sum().clamp_min(1e-30)
    student_energy = student.square().sum().clamp_min(1e-30)
    dot = (student * teacher).sum()
    return {
        "error_energy": float(error),
        "teacher_effect_energy": float(teacher_energy),
        "student_effect_energy": float(student_energy),
        "dot": float(dot),
    }


def fisher_effect_agreement(teacher_base, teacher_changed, student_base, student_changed):
    """Softmax-Fisher agreement between two finite logit interventions.

    The teacher's baseline distribution supplies the vocabulary weights. Each
    logit effect is centered under that distribution, quotienting out the
    constant-logit direction to which softmax is invariant.
    """
    weights = F.softmax(teacher_base.float(), dim=-1)
    teacher = (teacher_changed - teacher_base).float()
    student = (student_changed - student_base).float()
    teacher = teacher - (weights * teacher).sum(-1, keepdim=True)
    student = student - (weights * student).sum(-1, keepdim=True)
    error = (weights * (student - teacher).square()).sum()
    teacher_energy = (weights * teacher.square()).sum().clamp_min(1e-30)
    student_energy = (weights * student.square()).sum().clamp_min(1e-30)
    dot = (weights * student * teacher).sum()
    return {
        "error_energy": float(error),
        "teacher_effect_energy": float(teacher_energy),
        "student_effect_energy": float(student_energy),
        "dot": float(dot),
    }


def combine_effect_agreements(rows):
    totals = {key: sum(row[key] for row in rows) for key in rows[0]}
    teacher = max(totals["teacher_effect_energy"], 1e-30)
    student = max(totals["student_effect_energy"], 1e-30)
    return {
        "effect_fidelity": 1 - totals["error_energy"] / teacher,
        "effect_cosine": totals["dot"] / math.sqrt(teacher * student),
        "effect_gain": totals["dot"] / teacher,
        **totals,
    }


def symmetrized_input_inner(left_a, right_a, left_b, right_b):
    """Frobenius inner products between sym(l r^T) matrices."""
    return 0.5 * (
        (left_a @ left_b.T) * (right_a @ right_b.T)
        + (left_a @ right_b.T) * (right_a @ left_b.T)
    )


def cp_term_cosine(model_a, model_b, *, chunk=128):
    """Gauge- and swap-aware CP tensor-term cosine matrix.

    CPBilinear.factors normalizes each factor and leaves magnitude/sign in
    ``gain``. The returned signed cosine includes the gain signs; permutation,
    left/right swaps, and factor rescaling do not affect it.
    """
    la, ra, oa, ga = [value.detach().float() for value in model_a.factors()]
    lb, rb, ob, gb = [value.detach().float() for value in model_b.factors()]
    qa = torch.sqrt(0.5 * (1 + (la * ra).sum(1).square())).clamp_min(1e-12)
    qb = torch.sqrt(0.5 * (1 + (lb * rb).sum(1).square())).clamp_min(1e-12)
    sign_a, sign_b = ga.sign().where(ga != 0, 1), gb.sign().where(gb != 0, 1)
    pieces = []
    for start in range(0, len(la), chunk):
        stop = min(start + chunk, len(la))
        q_inner = symmetrized_input_inner(la[start:stop], ra[start:stop], lb, rb)
        q_cosine = q_inner / (qa[start:stop, None] * qb[None, :])
        output_cosine = oa[:, start:stop].T @ ob
        pieces.append(
            q_cosine * output_cosine
            * sign_a[start:stop, None] * sign_b[None, :]
        )
    return torch.cat(pieces)


def factor_stability_summary(similarity):
    absolute = similarity.abs()
    a_best_value, a_best_index = absolute.max(1)
    b_best_value, b_best_index = absolute.max(0)
    rows = torch.arange(len(a_best_index), device=similarity.device)
    mutual = b_best_index[a_best_index] == rows
    diagonal = similarity.diag() if similarity.shape[0] == similarity.shape[1] else None
    result = {
        "a_to_b_median_best_abs_cosine": float(a_best_value.median()),
        "b_to_a_median_best_abs_cosine": float(b_best_value.median()),
        "mutual_nearest_fraction": float(mutual.float().mean()),
        "mutual_median_abs_cosine": (
            float(a_best_value[mutual].median()) if mutual.any() else float("nan")
        ),
    }
    if diagonal is not None:
        result.update({
            "index_aligned_median_abs_cosine": float(diagonal.abs().median()),
            "index_aligned_mean_signed_cosine": float(diagonal.mean()),
        })
    return result


def linear_cka(first, second):
    """Linear centered-kernel alignment using feature-space products."""
    first = first.float() - first.float().mean(0, keepdim=True)
    second = second.float() - second.float().mean(0, keepdim=True)
    cross = first.T @ second
    numerator = cross.square().sum()
    denominator = torch.sqrt(
        (first.T @ first).square().sum() * (second.T @ second).square().sum()
    ).clamp_min(1e-30)
    return float(numerator / denominator)


def routing_usage(counts):
    counts = torch.as_tensor(counts, dtype=torch.float64)
    total = counts.sum().clamp_min(1e-30)
    probabilities = counts / total
    nonzero = probabilities > 0
    entropy = -(probabilities[nonzero] * probabilities[nonzero].log()).sum()
    return {
        "features_ever_used": int(nonzero.sum()),
        "effective_support_simpson": float(total.square() / counts.square().sum().clamp_min(1e-30)),
        "effective_support_entropy": float(entropy.exp()),
        "normalized_entropy": float(entropy / math.log(max(2, len(counts)))),
    }


def bilinear_operation_counts(dimension, native_width, rank, active=None):
    """Approximate scalar multiply counts for one token, excluding Top-K sort."""
    native = 3 * dimension * native_width
    dense = 3 * dimension * rank
    result = {
        "native_multiplies": native,
        "dense_student_multiplies": dense,
        "dense_fraction_of_native": dense / native,
    }
    if active is not None:
        # Current router evaluates both input projections for every factor and
        # only sparsifies the output projection. A hypothetical indexed router
        # could compute all three projections for active factors only.
        current = 2 * dimension * rank + dimension * active
        ideal = 3 * dimension * active
        result.update({
            "active_factors": active,
            "active_fraction": active / rank,
            "current_routed_multiplies": current,
            "current_routed_fraction_of_native": current / native,
            "ideal_indexed_multiplies": ideal,
            "ideal_indexed_fraction_of_native": ideal / native,
        })
    return result

"""Fidelity metrics for intact and circuit logits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.floating]


def _validate_logits(full: FloatArray, masked: FloatArray, answer_index: int) -> None:
    if full.ndim != 1 or masked.ndim != 1 or full.shape != masked.shape:
        raise ValueError("full and masked logits must be same-shaped 1D arrays")
    if full.size < 2:
        raise ValueError("at least two vocabulary logits are required")
    if answer_index < 0 or answer_index >= full.size:
        raise IndexError("answer_index is out of bounds")
    if not np.isfinite(full).all() or not np.isfinite(masked).all():
        raise ValueError("logits must be finite")


def answer_margin(logits: FloatArray, answer_index: int) -> float:
    """Return answer logit minus the largest non-answer logit."""
    non_answer = np.concatenate((logits[:answer_index], logits[answer_index + 1 :]))
    return float(logits[answer_index] - non_answer.max())


def _log_softmax(logits: FloatArray) -> FloatArray:
    shifted = logits - logits.max()
    return shifted - np.log(np.exp(shifted).sum())


def forward_kl(full_logits: FloatArray, masked_logits: FloatArray) -> float:
    """Compute D_KL(p_full || p_masked) stably from logits."""
    full_log_probs = _log_softmax(full_logits)
    masked_log_probs = _log_softmax(masked_logits)
    full_probs = np.exp(full_log_probs)
    return float(np.sum(full_probs * (full_log_probs - masked_log_probs)))


@dataclass(frozen=True)
class FidelityMetrics:
    intact_margin: float
    masked_margin: float
    margin_retention: float
    answer_preserved: bool
    output_kl: float

    def meets(self, alpha: float) -> bool:
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        return self.masked_margin >= alpha * self.intact_margin


def compute_fidelity(
    full_logits: FloatArray,
    masked_logits: FloatArray,
    answer_index: int,
) -> FidelityMetrics:
    full = np.asarray(full_logits, dtype=np.float64)
    masked = np.asarray(masked_logits, dtype=np.float64)
    _validate_logits(full, masked, answer_index)

    intact_margin = answer_margin(full, answer_index)
    masked_margin = answer_margin(masked, answer_index)
    if intact_margin == 0:
        margin_retention = float("nan")
    else:
        margin_retention = masked_margin / intact_margin

    return FidelityMetrics(
        intact_margin=intact_margin,
        masked_margin=masked_margin,
        margin_retention=margin_retention,
        answer_preserved=int(masked.argmax()) == answer_index,
        output_kl=forward_kl(full, masked),
    )

"""Importance ranking and fresh-forward evaluation for Stage A circuits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import Literal

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from .arithmetic import SemanticOutcome, semantic_outcome
from .hf_adapter import GatedMLPMaskAdapter


Method = Literal["random", "activation", "write_magnitude", "gradient_activation"]


@dataclass(frozen=True)
class MaskEvaluation:
    example_id: str
    method: str
    retained: int
    total: int
    retained_fraction: float
    baseline: str
    answer: int
    answer_token_id: int
    intact_margin: float
    masked_margin: float
    margin_retention: float
    output_kl: float
    predicted_token_id: int
    predicted_number: int | None
    exact: bool
    failure_kind: str
    numeric_distance: int | None
    numeric_probability: float
    expected_distance: float
    elapsed_seconds: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def answer_margin(logits: Tensor, answer_token_id: int) -> Tensor:
    answer = logits[..., answer_token_id]
    before = logits[..., :answer_token_id].amax(dim=-1)
    after = logits[..., answer_token_id + 1 :].amax(dim=-1)
    return answer - torch.maximum(before, after)


def importance_scores(
    adapter: GatedMLPMaskAdapter,
    input_ids: Tensor,
    answer_token_id: int,
    mask: Tensor | None = None,
    difference_baseline: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor], float]:
    """Return intact logits and four [layer, position, neuron] score tensors."""
    start = time.perf_counter()
    traced = adapter.forward(input_ids, mask, retain_activation_grads=True)
    margin = answer_margin(traced.logits[0].float(), answer_token_id)
    gradients = torch.autograd.grad(margin, traced.activations)
    activations = torch.stack([item[0].detach().float() for item in traced.activations])
    grads = torch.stack([item[0].detach().float() for item in gradients])
    write_norms = adapter.write_norms().to(activations.device)[:, None, :]
    scores = {
        "activation": activations.abs(),
        "write_magnitude": activations.abs() * write_norms,
        "gradient_activation": (activations * grads).abs(),
    }
    if difference_baseline is not None:
        reference = difference_baseline.to(device=activations.device, dtype=activations.dtype)
        if reference.shape != activations.shape:
            raise ValueError(
                f"difference baseline has shape {tuple(reference.shape)}, "
                f"expected {tuple(activations.shape)}"
            )
        scores["gradient_difference"] = ((activations - reference) * grads).abs()
    return traced.logits[0].detach().float(), scores, time.perf_counter() - start


def stable_descending_ranking(scores: Tensor) -> Tensor:
    return torch.argsort(scores.reshape(-1), descending=True, stable=True)


def random_ranking(total: int, seed: int, device: torch.device) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randperm(total, generator=generator).to(device)


def mask_from_ranking(
    ranking: Tensor,
    retained: int,
    shape: tuple[int, int, int],
    *,
    invert: bool = False,
) -> Tensor:
    if retained < 0 or retained > ranking.numel():
        raise ValueError("retained is outside the mask size")
    flat = torch.ones(ranking.numel(), device=ranking.device) if invert else torch.zeros(
        ranking.numel(), device=ranking.device
    )
    if retained:
        flat[ranking[:retained]] = 0.0 if invert else 1.0
    return flat.view(shape)


def _kl(full_logits: Tensor, masked_logits: Tensor) -> float:
    full_log = F.log_softmax(full_logits.float(), dim=-1)
    masked_log = F.log_softmax(masked_logits.float(), dim=-1)
    return float(torch.sum(full_log.exp() * (full_log - masked_log)))


def evaluate_mask(
    adapter: GatedMLPMaskAdapter,
    input_ids: Tensor,
    mask: Tensor,
    full_logits: Tensor,
    *,
    example_id: str,
    method: str,
    answer: int,
    answer_token_id: int,
    number_token_ids: dict[int, int],
    baseline: Tensor | None = None,
    baseline_name: str = "zero",
) -> MaskEvaluation:
    start = time.perf_counter()
    with torch.no_grad():
        masked_logits = adapter.forward(input_ids, mask, baseline=baseline).logits[0].float()
    elapsed = time.perf_counter() - start
    intact_margin = float(answer_margin(full_logits, answer_token_id))
    masked_margin = float(answer_margin(masked_logits, answer_token_id))
    outcome: SemanticOutcome = semantic_outcome(
        masked_logits.cpu().numpy(), answer, number_token_ids
    )
    retained = int(mask.sum().item())
    total = mask.numel()
    return MaskEvaluation(
        example_id=example_id,
        method=method,
        retained=retained,
        total=total,
        retained_fraction=retained / total,
        baseline=baseline_name,
        answer=answer,
        answer_token_id=answer_token_id,
        intact_margin=intact_margin,
        masked_margin=masked_margin,
        margin_retention=masked_margin / intact_margin if intact_margin else math.nan,
        output_kl=_kl(full_logits, masked_logits),
        predicted_token_id=outcome.predicted_token_id,
        predicted_number=outcome.predicted_number,
        exact=outcome.exact,
        failure_kind=outcome.failure_kind,
        numeric_distance=outcome.numeric_distance,
        numeric_probability=outcome.numeric_probability,
        expected_distance=outcome.expected_distance,
        elapsed_seconds=elapsed,
    )


def retained_counts(total: int, fractions: list[float]) -> list[int]:
    counts = {0, total}
    counts.update(max(1, min(total, round(total * fraction))) for fraction in fractions)
    return sorted(counts)


def smallest_feasible(
    evaluations: list[MaskEvaluation], alpha: float
) -> MaskEvaluation | None:
    feasible = [
        item
        for item in evaluations
        if item.masked_margin >= alpha * item.intact_margin
    ]
    return min(feasible, key=lambda item: item.retained) if feasible else None

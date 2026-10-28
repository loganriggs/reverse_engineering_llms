"""Family-aware semantic outcomes for language probe ablations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time

import torch
from torch import Tensor
import torch.nn.functional as F

from .extraction import answer_margin
from .hf_adapter import GatedMLPMaskAdapter
from .language_tasks import LanguageExample


@dataclass(frozen=True)
class LanguageMaskEvaluation:
    example_id: str
    family: str
    method: str
    retained: int
    total: int
    retained_fraction: float
    baseline: str
    answer_text: str
    answer_token_id: int
    intact_margin: float
    masked_margin: float
    margin_retention: float
    output_kl: float
    predicted_token_id: int
    predicted_text: str
    exact: bool
    failure_kind: str
    candidate_probability: float
    expected_semantic_cost: float
    elapsed_seconds: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def semantic_vocabulary(tokenizer, example: LanguageExample) -> tuple[dict[int, tuple[str, float]], float]:
    """Map relevant tokens to (category, cost); return non-candidate cost."""
    answer_id = tokenizer.encode(example.answer_text, add_special_tokens=False)
    if len(answer_id) != 1:
        raise ValueError(f"answer is not one token: {example.answer_text!r} -> {answer_id}")
    mapping: dict[int, tuple[str, float]] = {answer_id[0]: ("exact", 0.0)}
    if example.family == "ioi":
        from .language_tasks import NAMES
        contrast_id = tokenizer.encode(example.contrast_text, add_special_tokens=False)
        if len(contrast_id) == 1:
            mapping[contrast_id[0]] = ("role_reversal", 1.0)
        for name in NAMES:
            token_ids = tokenizer.encode(f" {name}", add_special_tokens=False)
            if len(token_ids) == 1 and token_ids[0] not in mapping:
                mapping[token_ids[0]] = ("other_name", 2.0)
        return mapping, 3.0
    if example.family == "agreement":
        contrast_id = tokenizer.encode(example.contrast_text, add_special_tokens=False)
        if len(contrast_id) != 1:
            raise ValueError("agreement contrast is not one token")
        mapping[contrast_id[0]] = ("wrong_agreement", 1.0)
        return mapping, 2.0
    for opener in (" (", " [", " {"):
        token_ids = tokenizer.encode(opener, add_special_tokens=False)
        if len(token_ids) == 1:
            mapping[token_ids[0]] = ("opener", 2.0)
    for closer in (" )\n", " ]\n", " }\n"):
        token_ids = tokenizer.encode(closer, add_special_tokens=False)
        if len(token_ids) == 1 and token_ids[0] not in mapping:
            mapping[token_ids[0]] = ("wrong_closer", 1.0)
    return mapping, 3.0


def evaluate_language_mask(
    adapter: GatedMLPMaskAdapter,
    tokenizer,
    example: LanguageExample,
    input_ids: Tensor,
    mask: Tensor,
    full_logits: Tensor,
    *,
    method: str,
    baseline: Tensor | None = None,
    baseline_name: str = "zero",
) -> LanguageMaskEvaluation:
    mapping, other_cost = semantic_vocabulary(tokenizer, example)
    answer_id = tokenizer.encode(example.answer_text, add_special_tokens=False)[0]
    start = time.perf_counter()
    with torch.no_grad():
        masked_logits = adapter.forward(input_ids, mask, baseline=baseline).logits[0].float()
    elapsed = time.perf_counter() - start
    full_log = F.log_softmax(full_logits.float(), dim=-1)
    masked_log = F.log_softmax(masked_logits, dim=-1)
    kl = float(torch.sum(full_log.exp() * (full_log - masked_log)))
    probabilities = masked_log.exp()
    candidate_probability = float(sum(probabilities[token_id] for token_id in mapping))
    expected_cost = sum(
        float(probabilities[token_id]) * cost
        for token_id, (_, cost) in mapping.items()
    ) + other_cost * (1.0 - candidate_probability)
    predicted_id = int(masked_logits.argmax())
    predicted_text = tokenizer.decode([predicted_id])
    category = mapping.get(predicted_id, ("other", other_cost))[0]
    intact_margin = float(answer_margin(full_logits, answer_id))
    masked_margin = float(answer_margin(masked_logits, answer_id))
    retained = int(mask.sum().item())
    return LanguageMaskEvaluation(
        example_id=example.example_id,
        family=example.family,
        method=method,
        retained=retained,
        total=mask.numel(),
        retained_fraction=retained / mask.numel(),
        baseline=baseline_name,
        answer_text=example.answer_text,
        answer_token_id=answer_id,
        intact_margin=intact_margin,
        masked_margin=masked_margin,
        margin_retention=masked_margin / intact_margin if intact_margin else math.nan,
        output_kl=kl,
        predicted_token_id=predicted_id,
        predicted_text=predicted_text,
        exact=category == "exact",
        failure_kind=category,
        candidate_probability=candidate_probability,
        expected_semantic_cost=expected_cost,
        elapsed_seconds=elapsed,
    )

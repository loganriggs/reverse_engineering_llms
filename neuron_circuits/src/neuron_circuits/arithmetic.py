"""Single-token arithmetic probes with graded semantic answer distances."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import numpy.typing as npt


Operation = Literal["add", "subtract", "multiply"]


@dataclass(frozen=True)
class ArithmeticExample:
    example_id: str
    prompt: str
    answer: int
    operation: Operation
    left: int
    right: int


def arithmetic_examples() -> list[ArithmeticExample]:
    """Return fixed-format, single-digit problems with answers from zero to nine."""
    examples: list[ArithmeticExample] = []
    for operation in ("add", "subtract", "multiply"):
        for left in range(10):
            for right in range(10):
                if operation == "add":
                    answer = left + right
                    symbol = "+"
                elif operation == "subtract":
                    answer = left - right
                    symbol = "-"
                else:
                    answer = left * right
                    symbol = "*"
                if 0 <= answer <= 9:
                    examples.append(
                        ArithmeticExample(
                            example_id=f"{operation}-{left}-{right}",
                            prompt=f"What is {left} {symbol} {right}? Answer: ",
                            answer=answer,
                            operation=operation,
                            left=left,
                            right=right,
                        )
                    )
    return examples


@dataclass(frozen=True)
class SemanticOutcome:
    predicted_token_id: int
    predicted_number: int | None
    exact: bool
    failure_kind: Literal["exact", "nearby", "far", "non_numeric"]
    numeric_distance: int | None
    numeric_probability: float
    expected_distance: float


def semantic_outcome(
    logits: npt.NDArray[np.floating],
    answer: int,
    number_token_ids: dict[int, int],
    *,
    nearby_radius: int = 1,
    non_numeric_cost: float = 10.0,
) -> SemanticOutcome:
    """Summarize exactness and how semantically far a masked answer moved.

    ``expected_distance`` uses absolute numeric error for number tokens and a
    stated fixed cost for all non-number vocabulary mass.
    """
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - values.max()
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    token_to_number = {token: number for number, token in number_token_ids.items()}
    predicted_token = int(values.argmax())
    predicted_number = token_to_number.get(predicted_token)
    numeric_probability = float(sum(probabilities[token] for token in token_to_number))
    expected = sum(
        float(probabilities[token]) * abs(number - answer)
        for token, number in token_to_number.items()
    ) + non_numeric_cost * (1.0 - numeric_probability)

    if predicted_number is None:
        kind: Literal["exact", "nearby", "far", "non_numeric"] = "non_numeric"
        distance = None
    else:
        distance = abs(predicted_number - answer)
        if distance == 0:
            kind = "exact"
        elif distance <= nearby_radius:
            kind = "nearby"
        else:
            kind = "far"
    return SemanticOutcome(
        predicted_token_id=predicted_token,
        predicted_number=predicted_number,
        exact=kind == "exact",
        failure_kind=kind,
        numeric_distance=distance,
        numeric_probability=numeric_probability,
        expected_distance=float(expected),
    )


def operation_counts(examples: Iterable[ArithmeticExample]) -> dict[str, int]:
    counts = {"add": 0, "subtract": 0, "multiply": 0}
    for example in examples:
        counts[example.operation] += 1
    return counts

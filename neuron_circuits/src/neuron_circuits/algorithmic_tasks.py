"""Controlled list algorithms and one-element counterfactuals."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Literal


Algorithm = Literal["maximum", "minimum", "second_largest", "argmax", "second_argmax"]


@dataclass(frozen=True)
class AlgorithmExample:
    example_id: str
    algorithm: Algorithm
    values: tuple[int, ...]
    prompt: str
    answer: int


@dataclass(frozen=True)
class AlgorithmCounterfactualPair:
    pair_id: str
    family: str
    base: AlgorithmExample
    source: AlgorithmExample
    changed_position: int


def answer_for(values: tuple[int, ...], algorithm: Algorithm) -> int:
    if len(set(values)) != len(values):
        raise ValueError("algorithm probes require distinct values")
    if algorithm == "maximum":
        return max(values)
    if algorithm == "minimum":
        return min(values)
    if algorithm == "second_largest":
        return sorted(values)[-2]
    if algorithm == "argmax":
        return values.index(max(values))
    if algorithm == "second_argmax":
        return values.index(sorted(values)[-2])
    raise ValueError(f"unknown algorithm {algorithm}")


def prompt_for(values: tuple[int, ...], algorithm: Algorithm) -> str:
    rendered = ", ".join(str(value) for value in values)
    labels = {
        "maximum": "maximum number",
        "minimum": "minimum number",
        "second_largest": "second largest number",
        "argmax": "0-based index of the maximum number",
        "second_argmax": "0-based index of the second largest number",
    }
    return f"Find the {labels[algorithm]} in [{rendered}]. Answer: "


def algorithm_examples(algorithm: Algorithm, *, length: int = 4) -> list[AlgorithmExample]:
    output = []
    for values in permutations(range(10), length):
        answer = answer_for(values, algorithm)
        slug = "-".join(map(str, values))
        output.append(AlgorithmExample(
            example_id=f"{algorithm}-{slug}",
            algorithm=algorithm,
            values=values,
            prompt=prompt_for(values, algorithm),
            answer=answer,
        ))
    return output


def algorithm_counterfactual_pairs(algorithm: Literal["maximum", "minimum"], *, length: int = 4) -> list[AlgorithmCounterfactualPair]:
    """Return aligned examples differing at one list position with changed answers."""
    examples = algorithm_examples(algorithm, length=length)
    lookup = {example.values: example for example in examples}
    output = []
    for base in examples:
        for position in range(length):
            for replacement in range(10):
                if replacement in base.values:
                    continue
                changed = list(base.values)
                changed[position] = replacement
                source = lookup[tuple(changed)]
                if source.answer == base.answer:
                    continue
                output.append(AlgorithmCounterfactualPair(
                    pair_id=f"{algorithm}-{'-'.join(map(str, base.values))}-p{position}-to-{replacement}",
                    family=algorithm,
                    base=base,
                    source=source,
                    changed_position=position,
                ))
    return output

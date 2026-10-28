"""Controlled single-token relational-retrieval prompts."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Callable, Iterable, Literal, Sequence


Variant = Literal["extraction", "query", "payload", "surface"]


NAMES = (
    "Ava", "Bea", "Cleo", "Dara", "Esme", "Faye", "Gail", "Hope",
    "Iris", "Jade", "Kira", "Lena", "Maya", "Nora", "Opal", "Pia",
)
PAYLOADS = ("amber", "blue", "coral", "green", "ivory", "lilac", "red", "teal")


@dataclass(frozen=True)
class Example:
    example_id: str
    group_id: str
    split: Literal["extraction", "transfer"]
    variant: Variant
    prompt: str
    answer: str
    query_name: str
    facts: tuple[tuple[str, str], ...]
    distractor: tuple[str, str]

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def _render(facts: tuple[tuple[str, str], ...], query_name: str) -> str:
    clauses = " ".join(f"{name} has {value}." for name, value in facts)
    return f"{clauses} What does {query_name} have? Answer:"


class RelationalRetrievalGenerator:
    """Generate balanced examples and aligned, changed-answer counterfactuals."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def _base(self, index: int, answer: str) -> Example:
        names = tuple(self._rng.sample(NAMES, 4))
        alternatives = [payload for payload in PAYLOADS if payload != answer]
        other_payloads = tuple(self._rng.sample(alternatives, 3))
        answer_pair = (names[0], answer)
        other_pairs = list(zip(names[1:], other_payloads, strict=True))
        distractor = other_pairs[-1]
        visible_pairs = [answer_pair, *other_pairs]
        self._rng.shuffle(visible_pairs)
        facts = tuple(visible_pairs)
        query_name = answer_pair[0]
        group_id = f"group-{index:04d}"
        return Example(
            example_id=f"{group_id}-base",
            group_id=group_id,
            split="extraction",
            variant="extraction",
            prompt=_render(facts, query_name),
            answer=answer,
            query_name=query_name,
            facts=facts,
            distractor=distractor,
        )

    def _counterfactual(self, base: Example, number: int) -> Example:
        variant: Variant = ("query", "payload", "surface")[number % 3]  # type: ignore[assignment]
        facts = list(base.facts)
        query_name = base.query_name

        if variant == "query":
            choices = [(name, payload) for name, payload in facts if name != query_name]
            query_name, answer = self._rng.choice(choices)
        elif variant == "payload":
            query_index = next(i for i, (name, _) in enumerate(facts) if name == query_name)
            old_answer = facts[query_index][1]
            candidates = [p for p in PAYLOADS if p not in {value for _, value in facts}]
            answer = self._rng.choice(candidates)
            facts[query_index] = (query_name, answer)
            assert answer != old_answer
        else:
            replacement_names = self._rng.sample(
                [name for name in NAMES if name not in {n for n, _ in facts}], len(facts)
            )
            mapping = {old: new for (old, _), new in zip(facts, replacement_names, strict=True)}
            facts = [(mapping[name], payload) for name, payload in facts]
            query_name = mapping[query_name]
            # A surface-only example checks invariance; changed-answer variants are
            # supplied by query and payload counterfactuals.
            answer = base.answer

        frozen_facts = tuple(facts)
        return Example(
            example_id=f"{base.group_id}-transfer-{number:02d}",
            group_id=base.group_id,
            split="transfer",
            variant=variant,
            prompt=_render(frozen_facts, query_name),
            answer=answer,
            query_name=query_name,
            facts=frozen_facts,
            distractor=base.distractor,
        )

    def generate(
        self,
        extraction_size: int = 100,
        transfer_per_example: int = 2,
    ) -> list[Example]:
        if extraction_size <= 0:
            raise ValueError("extraction_size must be positive")
        if transfer_per_example < 0:
            raise ValueError("transfer_per_example must be non-negative")

        examples: list[Example] = []
        for index in range(extraction_size):
            base = self._base(index, PAYLOADS[index % len(PAYLOADS)])
            examples.append(base)
            examples.extend(
                self._counterfactual(base, number)
                for number in range(transfer_per_example)
            )
        return examples


def answer_counts(examples: Iterable[Example]) -> Counter[str]:
    return Counter(example.answer for example in examples)


def validate_token_alignment(
    examples: Iterable[Example], encode: Callable[[str], Sequence[int]]
) -> dict[str, int]:
    """Return one token length per group, rejecting misaligned transfer prompts."""
    lengths: dict[str, int] = {}
    for example in examples:
        length = len(encode(example.prompt))
        prior = lengths.setdefault(example.group_id, length)
        if length != prior:
            raise ValueError(
                f"token length mismatch in {example.group_id}: expected {prior}, got {length}"
            )
    return lengths


def write_jsonl(examples: Iterable[Example], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(example.to_json() + "\n")

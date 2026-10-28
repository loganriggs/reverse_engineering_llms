"""Controlled single-token IOI, agreement, and bracket-closing probes."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations
from typing import Literal


Family = Literal["ioi", "agreement", "brackets"]


@dataclass(frozen=True)
class LanguageExample:
    example_id: str
    family: Family
    prompt: str
    answer_text: str
    answer_label: str
    contrast_text: str | None = None


@dataclass(frozen=True)
class CounterfactualPair:
    pair_id: str
    family: Family
    base: LanguageExample
    source: LanguageExample
    change: str


NAMES = (
    "John", "Mary", "Alice", "Bob", "Carol", "Dave",
    "Sarah", "Tom", "Emma", "James", "Laura", "Robert",
)


def ioi_examples() -> list[LanguageExample]:
    examples = []
    for first, second in combinations(NAMES, 2):
        for subject, answer in ((first, second), (second, first)):
            examples.append(
                LanguageExample(
                    example_id=f"ioi-{first.lower()}-{second.lower()}-{subject.lower()}",
                    family="ioi",
                    prompt=(
                        f"{first} and {second} went to the store. "
                        f"{subject} gave the book to"
                    ),
                    answer_text=f" {answer}",
                    answer_label=answer,
                    contrast_text=f" {subject}",
                )
            )
    return examples


NOUNS = (
    ("key", "keys"),
    ("dog", "dogs"),
    ("cat", "cats"),
    ("report", "reports"),
    ("picture", "pictures"),
    ("bottle", "bottles"),
    ("tree", "trees"),
    ("book", "books"),
)


def agreement_examples() -> list[LanguageExample]:
    examples = []
    for subject_index, (singular, plural) in enumerate(NOUNS):
        for attractor_index, (other_singular, other_plural) in enumerate(NOUNS):
            if subject_index == attractor_index:
                continue
            for subject, attractor, answer, contrast, number in (
                (singular, other_plural, " is", " are", "singular"),
                (plural, other_singular, " are", " is", "plural"),
            ):
                examples.append(
                    LanguageExample(
                        example_id=f"agreement-{subject}-{attractor}",
                        family="agreement",
                        prompt=f"Complete with is or are: The {subject} near the {attractor}",
                        answer_text=answer,
                        answer_label=number,
                        contrast_text=contrast,
                    )
                )
    return examples


BRACKET_PREFIX = "Complete with one closing bracket.\n( -> )\n[ -> ]\n{ -> }\n"
BALANCED_INNERS = (
    "()", "[]", "{}", "(())", "([])", "({})",
    "[()]", "[[]]", "[{}]", "{()}", "{[]}", "{{}}",
)
BRACKETS = (("(", ")"), ("[", "]"), ("{", "}"))


def bracket_examples() -> list[LanguageExample]:
    examples = []
    for opener, closer in BRACKETS:
        for index, inner in enumerate(BALANCED_INNERS):
            sequence = opener + inner
            examples.append(
                LanguageExample(
                    example_id=f"brackets-{ord(opener)}-{index}",
                    family="brackets",
                    prompt=f"{BRACKET_PREFIX}{sequence} ->",
                    answer_text=f" {closer}\n",
                    answer_label=closer,
                )
            )
    return examples


def all_language_examples() -> list[LanguageExample]:
    return [*ioi_examples(), *agreement_examples(), *bracket_examples()]


def language_counterfactual_pairs() -> list[CounterfactualPair]:
    """Minimal prompt pairs whose correct next-token answers differ."""
    pairs: list[CounterfactualPair] = []
    for first, second in combinations(NAMES, 2):
        base = LanguageExample(
            f"ioi-{first.lower()}-{second.lower()}-{first.lower()}", "ioi",
            f"{first} and {second} went to the store. {first} gave the book to",
            f" {second}", second, f" {first}",
        )
        source = LanguageExample(
            f"ioi-{first.lower()}-{second.lower()}-{second.lower()}", "ioi",
            f"{first} and {second} went to the store. {second} gave the book to",
            f" {first}", first, f" {second}",
        )
        pairs.extend((
            CounterfactualPair(
                f"ioi-{first.lower()}-{second.lower()}-forward", "ioi", base, source,
                "swap which person gives the object",
            ),
            CounterfactualPair(
                f"ioi-{first.lower()}-{second.lower()}-reverse", "ioi", source, base,
                "swap which person gives the object",
            ),
        ))

    for subject_index, (singular, plural) in enumerate(NOUNS):
        for attractor_index, (other_singular, other_plural) in enumerate(NOUNS):
            if subject_index == attractor_index:
                continue
            base = LanguageExample(
                f"agreement-{singular}-{other_plural}", "agreement",
                f"Complete with is or are: The {singular} near the {other_plural}",
                " is", "singular", " are",
            )
            source = LanguageExample(
                f"agreement-{plural}-{other_singular}", "agreement",
                f"Complete with is or are: The {plural} near the {other_singular}",
                " are", "plural", " is",
            )
            pairs.extend((
                CounterfactualPair(
                    f"agreement-{singular}-{other_singular}-sg-to-pl", "agreement", base, source,
                    "flip subject number while keeping the attractor opposite",
                ),
                CounterfactualPair(
                    f"agreement-{singular}-{other_singular}-pl-to-sg", "agreement", source, base,
                    "flip subject number while keeping the attractor opposite",
                ),
            ))

    for inner_index, inner in enumerate(BALANCED_INNERS):
        for (base_opener, base_closer), (source_opener, source_closer) in permutations(BRACKETS, 2):
            base = LanguageExample(
                f"brackets-{ord(base_opener)}-{inner_index}", "brackets",
                f"{BRACKET_PREFIX}{base_opener + inner} ->", f" {base_closer}\n", base_closer,
            )
            source = LanguageExample(
                f"brackets-{ord(source_opener)}-{inner_index}", "brackets",
                f"{BRACKET_PREFIX}{source_opener + inner} ->", f" {source_closer}\n", source_closer,
            )
            pairs.append(CounterfactualPair(
                f"brackets-{ord(base_opener)}-to-{ord(source_opener)}-{inner_index}",
                "brackets", base, source, "change the unmatched outer opener",
            ))
    return pairs

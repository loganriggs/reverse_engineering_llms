"""Empirical biography paragraphs with a controlled subject-name reuse rule.

The base text is sampled from WikiBio. This DGP only claims the explicit
two-mention surname relation, not an oracle for every biography fact.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from audit_wikibio_document_rules import (
    SUBJECT_VERBS, candidate, field_map, mentions,
)


@dataclass(frozen=True)
class Arm:
    text: str
    surname: str
    target_span: tuple[int, int]


@dataclass(frozen=True)
class Pair:
    original: Arm
    edited: Arm


def visible_answer(text):
    """Parse only the visible first name and next subject-initial surname."""
    lines = text.strip().lower().splitlines()
    if len(lines) < 2:
        raise ValueError('No later sentence')
    first_words = re.findall(r"[^\W\d_]+(?:-[^\W\d_]+)*", lines[0])[:4]
    for surname in reversed(first_words):
        if len(surname) < 4 or not surname.isalpha():
            continue
        later = [line for line in lines[1:]
                 if line.startswith(surname + ' ') and
                 (words := re.findall(r"[^\W\d_]+", line[len(surname):])) and
                 words[0] in SUBJECT_VERBS]
        if len(later) == 1:
            return surname
    raise ValueError('No unique visible subject reuse')


def make_pair(row, replacement):
    site = candidate(row)
    if site is None or not site['strict_editable']:
        raise ValueError('Source does not meet strict two-mention screen')
    original = site['text']
    surname = site['surname']
    replacement = replacement.lower().strip()
    if (not replacement.isalpha() or len(replacement) < 4 or
            replacement == surname or mentions(original, replacement)):
        raise ValueError('Unsafe replacement surname')
    spans = mentions(original, surname)
    if len(spans) != 2:
        raise ValueError('Source changed after screen')
    edited = original
    for span in reversed(spans):
        edited = edited[:span.start()] + replacement + edited[span.end():]
    edited_spans = mentions(edited, replacement)
    if len(edited_spans) != 2:
        raise ValueError('Replacement not unique')
    if visible_answer(original) != surname or visible_answer(edited) != replacement:
        raise ValueError('Visible rule did not change as intended')
    fields = field_map(row)
    name = fields.get('name', fields.get('article_title', ''))
    if not original.startswith(name):
        raise ValueError('Name mismatch')
    return Pair(
        Arm(original, surname, spans[1].span()),
        Arm(edited, replacement, edited_spans[1].span()),
    )

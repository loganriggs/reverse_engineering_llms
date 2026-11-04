"""Sample an arbitrary repeated three-letter code in varied web prose.

This is a literal surface-copy task, not an entity-binding oracle. Replacing
an acronym may change the truth or naturalness of the surrounding document.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
import re


CODE = re.compile(r'(?<![A-Za-z])[A-Z]{3}(?![A-Za-z])')
WORD = re.compile(r"\b[A-Za-z]+(?:'[A-Za-z]+)?\b")
# This bank contains familiar acronym-like BPE tokens rather than arbitrary
# uppercase English syllables. It is a surface prior, not an entity-type model.
ACRONYM_CODES = (
    'API', 'CPU', 'GPU', 'SQL', 'USB', 'RAM', 'LAN', 'VPN', 'URL', 'PDF',
    'AMD', 'ABC', 'BBC', 'CBS', 'CNN', 'NPR', 'NBC', 'FOX', 'AFP',
    'NBA', 'NFL', 'UFC', 'CIA', 'FBI', 'NSA', 'EPA', 'CDC', 'WHO',
    'ETF', 'USD', 'BTC', 'SEC', 'FAQ', 'AAA')
SOURCE_EXCLUDE = {
    'THE', 'AND', 'FOR', 'YOU', 'ALL', 'ANY', 'ONE', 'NEW', 'YES',
    'NOT', 'BUT', 'OUR', 'WAY', 'END', 'MEN', 'ARE', 'WHY', 'HOW',
    'CAN', 'NOW', 'GET', 'TOP', 'SET', 'ITS', 'SEE', 'SAY', 'USE',
}


@dataclass(frozen=True)
class Plan:
    source_text: str
    source_code: str
    spans: tuple[tuple[int, int], ...]
    target_occurrence: int
    gap_words: int


@dataclass(frozen=True)
class Realization:
    text: str
    code: str
    spans: tuple[tuple[int, int], ...]
    target_occurrence: int
    plan: Plan

    @property
    def answer_span(self):
        return self.spans[self.target_occurrence]

    @property
    def previous_span(self):
        return self.spans[self.target_occurrence - 1]


def candidates(text, min_gap_words=20, max_gap_words=160):
    by_code = {}
    for match in CODE.finditer(text):
        if match.group() not in SOURCE_EXCLUDE:
            by_code.setdefault(match.group(), []).append(match.span())
    for code, spans in by_code.items():
        if len(spans) < 2:
            continue
        for target in range(1, len(spans)):
            gap = len(WORD.findall(text[spans[target - 1][1]:spans[target][0]]))
            if min_gap_words <= gap <= max_gap_words:
                yield Plan(text, code, tuple(spans), target, gap)


def realize(plan, replacement):
    if not re.fullmatch('[A-Z]{3}', replacement):
        raise ValueError('Replacement must be three uppercase ASCII letters')
    if replacement == plan.source_code:
        raise ValueError('Replacement/source invalid')
    if re.search(r'(?<![A-Za-z])' + re.escape(replacement) +
                 r'(?![A-Za-z])', plan.source_text):
        raise ValueError('Replacement already occurs in source')
    pieces, cursor = [], 0
    for start, end in plan.spans:
        assert plan.source_text[start:end] == plan.source_code
        pieces.extend((plan.source_text[cursor:start], replacement))
        cursor = end
    pieces.append(plan.source_text[cursor:])
    result = ''.join(pieces)
    assert len(result) == len(plan.source_text)
    for start, end in plan.spans:
        assert result[start:end] == replacement
    return Realization(result, replacement, plan.spans,
                       plan.target_occurrence, plan)


def codebook(tokenizer):
    """Same-length standalone codes with and without a leading-space BPE."""
    vocab = tokenizer.get_vocab()
    return tuple(code for code in ACRONYM_CODES
                 if code in vocab and 'Ġ' + code in vocab)


def draw_pair(plan, tokenizer, rng: random.Random):
    pool = [code for code in codebook(tokenizer)
            if code != plan.source_code and
            not re.search(r'(?<![A-Za-z])' + re.escape(code) +
                          r'(?![A-Za-z])', plan.source_text)]
    if len(pool) < 2:
        return None
    left, right = rng.sample(pool, 2)
    return realize(plan, left), realize(plan, right)

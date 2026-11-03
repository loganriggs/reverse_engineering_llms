"""Independent spelling-style and bracket rules in one FineWeb document.

The document is sampled from human web text. Only two selected rule sites
have known conditional answers; this is not a full-text symbolic oracle.
"""

from __future__ import annotations

from dataclasses import dataclass
import random

from corpus_transformed_delimiters import candidates as delimiter_candidates
from corpus_transformed_spelling_style import (
    candidate_indices, transform as style_transform,
)


@dataclass(frozen=True)
class Plan:
    source_text: str
    style_occurrence_index: int
    delimiter_pair_index: int
    joint_span_characters: int


@dataclass(frozen=True)
class Site:
    kind: str
    start: int
    end: int
    answer: str
    source_start: int
    source_end: int


@dataclass(frozen=True)
class Document:
    text: str
    sites: tuple[Site, Site]
    plan: Plan


def plans(text, max_joint_span=1200):
    words, style_options = candidate_indices(text)
    brackets = delimiter_candidates(text)
    choices = []
    for style_index, _ in style_options:
        cue, target = words[style_index-1:style_index+1]
        for bracket_index, (opener, closer, _) in enumerate(brackets):
            span = max(target.end, closer)-min(cue.start, opener)
            if span <= max_joint_span:
                choices.append(Plan(text, style_index, bracket_index, span))
    return tuple(choices)


def sample_plan(text, rng: random.Random, max_joint_span=1200):
    options = plans(text, max_joint_span)
    return rng.choice(options) if options else None


def _mapped_position(pos, style_example):
    shift = 0
    for old, new in zip(style_example.source_occurrences,
                        style_example.transformed_spans):
        if old.end <= pos:
            shift += (new[1]-new[0])-(old.end-old.start)
        elif old.start < pos < old.end:
            raise ValueError('Delimiter inside style word')
    return pos+shift


def realize(plan: Plan, style='us', bracket='round'):
    if style not in ('us', 'uk') or bracket not in ('round', 'square'):
        raise ValueError('Invalid arm')
    example = style_transform(
        plan.source_text, style,
        target_occurrence_index=plan.style_occurrence_index)
    assert example is not None
    pair = delimiter_candidates(plan.source_text)[plan.delimiter_pair_index]
    opener, closer = map(lambda p: _mapped_position(p, example), pair[:2])
    assert example.text[opener] == '(' and example.text[closer] == ')'
    left, right = ('(', ')') if bracket == 'round' else ('[', ']')
    text = (example.text[:opener]+left+example.text[opener+1:closer]+
            right+example.text[closer+1:])
    cue = example.cue_span
    target = example.target_span
    style_site = Site('style', *target, example.target_word, *cue)
    bracket_site = Site('bracket', closer, closer+1, right,
                        opener, opener+1)
    assert text[style_site.start:style_site.end] == example.target_word
    assert text[bracket_site.start:bracket_site.end] == right
    assert text[cue[0]:cue[1]] in (example.source_occurrences[
        plan.style_occurrence_index-1].us,
        example.source_occurrences[plan.style_occurrence_index-1].uk)
    return Document(text, (style_site, bracket_site), plan)


def validate_factorial(plan):
    arms = {(style, bracket): realize(plan, style, bracket)
            for style in ('us', 'uk') for bracket in ('round', 'square')}
    assert len({arms[style, bracket].sites[0].answer
                for style in ('us', 'uk') for bracket in ('round', 'square')}) == 2
    assert len({arms[style, bracket].sites[1].answer
                for style in ('us', 'uk') for bracket in ('round', 'square')}) == 2
    for style in ('us', 'uk'):
        a, b = arms[style, 'round'], arms[style, 'square']
        opener = b.sites[1].source_start
        closer = b.sites[1].start
        assert a.text[:opener] == b.text[:opener]
        assert a.text[opener] == '(' and b.text[opener] == '['
        assert a.text[opener+1:closer] == b.text[opener+1:closer]
        assert a.text[closer] == ')' and b.text[closer] == ']'
        assert a.text[closer+1:] == b.text[closer+1:]
        assert a.sites[0] == b.sites[0]
    for bracket in ('round', 'square'):
        a, b = arms['us', bracket], arms['uk', bracket]
        assert a.sites[1].answer == b.sites[1].answer
        assert a.sites[0].answer != b.sites[0].answer
        assert a.sites[0].source_end <= a.sites[0].start
        assert b.sites[0].source_end <= b.sites[0].start
    return arms

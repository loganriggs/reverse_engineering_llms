"""Winning-margin arithmetic and bracket closure in one natural report.

The first score sentence and final margin sentence have explicit semantics;
the intervening report text is a screened human-written carrier. Its
parenthetical supplies a second, independently editable rule site. There
is no complete next-token oracle for the carrier.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import random

from corpus_score_sentence_renderer import render
from sports_alias_bracket_dgp import bracket_pairs, visible_bracket_answer
from sports_document_dgp import _detokenize
from sports_entity_alias_dgp import _render_tokens
from sports_report_carrier_dgp import (
    Plan as ScorePlan, increment_winning_score,
    sample_plan as sample_score_plan, visible_answer)


@dataclass(frozen=True)
class Plan:
    score: ScorePlan
    opener_token: int
    closer_token: int


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


def sample_plan(rng: random.Random, group='train', carrier_words=100,
                query_kind='margin'):
    for _ in range(500):
        score = sample_score_plan(rng, group=group,
                                  carrier_words=carrier_words,
                                  query_kind=query_kind)
        pairs = bracket_pairs(score.carrier)
        if not pairs:
            continue
        opener, closer = rng.choice(pairs)
        plan = Plan(score, opener, closer)
        try:
            for score_arm in ('base', 'increment'):
                for bracket_arm in ('round', 'square'):
                    realize(plan, score_arm, bracket_arm)
        except (ValueError, AssertionError):
            continue
        return plan
    raise RuntimeError('No score report with a valid bracket site')


def realize(plan: Plan, score_arm='base', bracket_arm='round'):
    if score_arm not in ('base', 'increment') or \
            bracket_arm not in ('round', 'square'):
        raise ValueError('Invalid arm')
    score = plan.score if score_arm == 'base' else \
        increment_winning_score(plan.score)
    tokens = list(score.carrier)
    assert tokens[plan.opener_token] == '(' and \
        tokens[plan.closer_token] == ')'
    if bracket_arm == 'square':
        tokens[plan.opener_token] = '['
        tokens[plan.closer_token] = ']'
    carrier, spans = _render_tokens(tokens)
    assert carrier == _detokenize(tokens).replace('*NEWPARAGRAPH*', '\n\n')
    result = _detokenize(render(score.score_template, score.winner,
                                score.loser, score.high, score.low))
    marker = '{MARGIN}' if score.query_kind == 'margin' else '{SCORE}'
    assert score.query_form.count(marker) == 1
    before, after = score.query_form.split(marker)
    answer = str(score.high-score.low) if score.query_kind == 'margin' \
        else str(score.high)
    prefix = result+' '+carrier+' '+before
    text = prefix+answer+after
    closer = ')' if bracket_arm == 'round' else ']'
    offset = len(result)+1
    bracket_site = Site('bracket',
                        offset+spans[plan.closer_token][0],
                        offset+spans[plan.closer_token][1], closer,
                        offset+spans[plan.opener_token][0],
                        offset+spans[plan.opener_token][1])
    score_site = Site(score.query_kind, len(prefix),
                      len(prefix)+len(answer), answer, 0, len(result))
    assert text[bracket_site.start:bracket_site.end] == closer
    assert text[score_site.start:score_site.end] == answer
    assert visible_bracket_answer(text[:bracket_site.start]) == closer
    assert visible_answer(text[:score_site.start], score.query_kind) == answer
    return Document(text, (bracket_site, score_site), plan)

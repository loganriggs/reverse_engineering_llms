"""Two independently editable rules in the same human sports report.

Entity aliasing uses a full-name to later surname candidate. Bracket closure
uses an observed parenthetical, optionally changed to square brackets. The
original article is empirical, so only these sites have controlled rules.
"""

from __future__ import annotations

from dataclasses import dataclass
import random

from sports_entity_alias_dgp import (
    Plan as AliasPlan, _render_tokens, sample_plan as sample_alias_plan,
)


@dataclass(frozen=True)
class Plan:
    alias: AliasPlan
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


def bracket_pairs(tokens, max_closer=250, min_gap=3, max_gap=30):
    stack, pairs = [], []
    for index, word in enumerate(tokens[:max_closer]):
        if word == '(':
            stack.append(index)
        elif word == ')' and stack:
            opener = stack.pop()
            if min_gap <= index-opener <= max_gap and \
                    not any(token in ('(', ')')
                            for token in tokens[opener+1:index]):
                pairs.append((opener, index))
    return tuple(pairs)


def sample_plan(rng: random.Random, group='train', alias_prior='roster',
                alias_exponent=0.):
    for _ in range(500):
        alias = sample_alias_plan(rng, group, alias_prior=alias_prior,
                                  alias_exponent=alias_exponent)
        pairs = bracket_pairs(alias.source_tokens)
        if pairs:
            opener, closer = rng.choice(pairs)
            return Plan(alias, opener, closer)
    raise RuntimeError('No report with both natural task sites')


def realize(plan: Plan, alias_arm='a', bracket_arm='round'):
    if alias_arm not in ('a', 'b') or bracket_arm not in ('round', 'square'):
        raise ValueError('Invalid rule arm')
    base = plan.alias
    alias = base.alias_a if alias_arm == 'a' else base.alias_b
    tokens = [alias if token == base.original_surname else token
              for token in base.source_tokens]
    assert tokens[plan.opener_token] == '(' and \
        tokens[plan.closer_token] == ')'
    if bracket_arm == 'square':
        tokens[plan.opener_token] = '['
        tokens[plan.closer_token] = ']'
    text, spans = _render_tokens(tokens)
    assert tokens[base.source_token:base.source_token+2] == \
        [base.first, alias]
    assert tokens[base.target_token] == alias
    alias_site = Site('alias', *spans[base.target_token], alias,
                      spans[base.source_token][0],
                      spans[base.source_token+1][1])
    closer = ')' if bracket_arm == 'round' else ']'
    bracket_site = Site('bracket', *spans[plan.closer_token], closer,
                        *spans[plan.opener_token])
    assert text[alias_site.start:alias_site.end] == alias
    assert text[bracket_site.start:bracket_site.end] == closer
    return Document(text, (alias_site, bracket_site), plan)


def visible_bracket_answer(prefix: str):
    """Read the latest unmatched opener from visible text only."""
    stack = []
    for char in prefix:
        if char in '([':
            stack.append(char)
        elif char in ')]':
            if not stack or (stack[-1], char) not in (('(', ')'), ('[', ']')):
                raise ValueError('Unbalanced visible bracket prefix')
            stack.pop()
    if not stack:
        raise ValueError('No visible opener for closer site')
    return ')' if stack[-1] == '(' else ']'

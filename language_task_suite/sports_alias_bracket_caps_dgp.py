"""Three independently editable rules in one human sports report.

The third rule uses a naturally occurring sentence boundary followed by a
common capitalized word. A period keeps its capitalized form; a semicolon
joins two clauses and lowercases that word. This is a controlled local
orthography edit, distinct from distant surname reuse and bracket closure.
The surrounding report is empirical and has no full symbolic LM oracle.
"""

from __future__ import annotations

from dataclasses import dataclass
import random

from sports_alias_bracket_dgp import (
    Plan as PairPlan,
    sample_plan as sample_pair_plan, visible_bracket_answer)
from sports_entity_alias_dgp import _render_tokens


CAP_WORDS = frozenset(('The', 'He', 'They', 'It', 'This', 'That',
                       'A', 'An'))


@dataclass(frozen=True)
class Plan:
    pair: PairPlan
    boundary_token: int


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
    sites: tuple[Site, Site, Site]
    plan: Plan


def caps_candidates(pair: PairPlan, max_target=250):
    base = pair.alias
    tokens = base.source_tokens
    out = []
    for i in range(8, min(len(tokens)-8, max_target)):
        if tokens[i] != '.' or tokens[i+1] not in CAP_WORDS:
            continue
        if i+1 in (base.source_token, base.source_token+1,
                   base.target_token, pair.opener_token,
                   pair.closer_token):
            continue
        if pair.opener_token < i+1 < pair.closer_token:
            continue
        if not tokens[i-1].isalpha() and tokens[i-1] != ')':
            continue
        out.append(i)
    return tuple(out)


def sample_plan(rng: random.Random, group='train', **kwargs):
    for _ in range(500):
        pair = sample_pair_plan(rng, group, **kwargs)
        options = caps_candidates(pair)
        if options:
            candidate = Plan(pair, rng.choice(options))
            try:
                realize(candidate, 'a', 'round', 'period')
                realize(candidate, 'a', 'square', 'period')
            except (ValueError, AssertionError):
                # A few source reports contain unbalanced unrelated marks.
                # Their visible bracket state cannot be certified.
                continue
            return candidate
    raise RuntimeError('No report with all three natural rule sites')


def visible_caps_answer(prefix, candidates):
    """Infer the target case from visible punctuation, not the sampled arm."""
    if len(candidates) != 2 or candidates[0].lower() != candidates[1]:
        raise ValueError('Expected capitalized/lowercase candidate pair')
    last = prefix.rstrip()[-1]
    if last == '.':
        return candidates[0]
    if last == ';':
        return candidates[1]
    raise ValueError('No supported sentence boundary before target')


def realize(plan: Plan, alias_arm='a', bracket_arm='round',
            caps_arm='period'):
    if alias_arm not in ('a', 'b') or bracket_arm not in ('round', 'square') \
            or caps_arm not in ('period', 'semicolon'):
        raise ValueError('Invalid factorial arm')
    pair, base = plan.pair, plan.pair.alias
    alias = base.alias_a if alias_arm == 'a' else base.alias_b
    tokens = [alias if word == base.original_surname else word
              for word in base.source_tokens]
    assert tokens[pair.opener_token] == '(' and \
        tokens[pair.closer_token] == ')'
    assert tokens[plan.boundary_token] == '.' and \
        tokens[plan.boundary_token+1] in CAP_WORDS
    if bracket_arm == 'square':
        tokens[pair.opener_token] = '['
        tokens[pair.closer_token] = ']'
    if caps_arm == 'semicolon':
        tokens[plan.boundary_token] = ';'
        tokens[plan.boundary_token+1] = \
            tokens[plan.boundary_token+1].lower()
    text, spans = _render_tokens(tokens)
    assert tokens[base.source_token:base.source_token+2] == \
        [base.first, alias]
    assert tokens[base.target_token] == alias
    alias_site = Site('alias', *spans[base.target_token], alias,
                      spans[base.source_token][0],
                      spans[base.source_token+1][1])
    closer = ')' if bracket_arm == 'round' else ']'
    bracket_site = Site('bracket', *spans[pair.closer_token], closer,
                        *spans[pair.opener_token])
    cap = tokens[plan.boundary_token+1]
    caps_site = Site('capitalization', *spans[plan.boundary_token+1], cap,
                     *spans[plan.boundary_token])
    assert all(text[site.start:site.end] == site.answer
               for site in (alias_site, bracket_site, caps_site))
    assert visible_bracket_answer(text[:bracket_site.start]) == closer
    assert visible_caps_answer(text[:caps_site.start],
                               (base.source_tokens[plan.boundary_token+1],
                                base.source_tokens[plan.boundary_token+1].lower())) == cap
    return Document(text, (alias_site, bracket_site, caps_site), plan)

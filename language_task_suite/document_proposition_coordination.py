"""A reversible coordination operator over proposition-surface V3 clauses.

This is a structural pilot. It joins adjacent complete, unscored assertions
without changing their contents, then validates the clauses by expanding the
coordination from *visible tokens only* before the existing semantic parser.
"""

from __future__ import annotations

import random

from document_proposition_surface import (
    AnswerSite, RenderedDocument, _opening, answer_from_visible_prefix,
    parse_visible_document, realize,
)


def _boundaries(tokens, sites, opening_length, rng, rate):
    periods = [i for i, token in enumerate(tokens) if token == '.']
    site_positions = {site.index for site in sites}
    chosen = []
    last_end = -1
    for k in range(1, len(periods)-1):
        left_end, right_end = periods[k], periods[k+1]
        left_start = periods[k-1]+1
        if left_end < opening_length or left_start <= last_end:
            continue
        # A newline denotes a paragraph boundary; do not combine across it.
        if left_end+1 >= len(tokens) or tokens[left_end+1] == '\n':
            continue
        if any(left_start <= pos <= right_end for pos in site_positions):
            continue
        if tokens[left_end+1] in ('.', ',', ';', ':', '?', '!'):
            continue
        if rng.random() < rate:
            chosen.append(left_end)
            last_end = right_end
    return tuple(chosen)


def compose(doc: RenderedDocument, seed=0, rate=.75):
    """Return (document, join_count), preserving all marked site answers."""
    if not 0 <= rate <= 1:
        raise ValueError(rate)
    tokens = doc.tokens
    if any(tokens[i:i+2] == (',', 'and') for i in range(len(tokens)-1)):
        raise ValueError('The reserved coordination marker is already present')
    joins = set(_boundaries(tokens, doc.sites, len(_opening(doc.plan)),
                            random.Random(seed), rate))
    result, indices = [], {}
    after_join = False
    for i, token in enumerate(tokens):
        indices[i] = len(result)
        if i in joins:
            result.extend((',', 'and'))
            after_join = True
        else:
            result.append(token.lower() if after_join and token in (
                'The', 'A', 'An', 'Now', 'In', 'For', 'According',
                'Part', 'Record', 'Count', 'There', 'As') else token)
            if after_join:
                after_join = False
    sites = tuple(AnswerSite(indices[site.index], site.family,
                             site.answer, site.candidates, site.entity)
                  for site in doc.sites)
    composed = RenderedDocument(tuple(result), sites, doc.form, doc.plan)
    for site in sites:
        assert composed.tokens[site.index] == site.answer
    return composed, len(joins)


def expand_visible(tokens):
    """Recover the original clause sequence without consulting a plan."""
    expanded = []
    i = 0
    while i < len(tokens):
        if tokens[i:i+2] == (',', 'and'):
            if i+2 >= len(tokens):
                raise ValueError('Trailing coordination marker')
            expanded.append('.')
            # The lowercase start is a rendering convention after comma-and;
            # the V3 parser expects the original sentence-initial case.
            start = tokens[i+2]
            expanded.append(start.capitalize() if start in (
                'the', 'a', 'an', 'now', 'in', 'for', 'according',
                'part', 'record', 'count', 'there', 'as') else start)
            i += 3
        else:
            expanded.append(tokens[i])
            i += 1
    return tuple(expanded)


def parse_composed(doc: RenderedDocument, lexical_enabled=False):
    return parse_visible_document(expand_visible(doc.tokens), lexical_enabled)


def answer_from_composed_prefix(prefix, lexical_enabled=False):
    return answer_from_visible_prefix(expand_visible(prefix), lexical_enabled)


def realize_composed(plan, seed=0, coordination_rate=.75,
                     lexical_exponent=None):
    base = realize(plan, seed=seed, lexical_exponent=lexical_exponent,
                   surface_version=3)
    return compose(base, seed=seed ^ 0xC005E, rate=coordination_rate)

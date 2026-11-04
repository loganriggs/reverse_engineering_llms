"""Truth-checked relative clauses attached to scored proposition statements.

The inserted clause states a *different*, already-visible attribute of the
same entity. Its truth is checked from preceding text, without reading the
sampled plan. This is a compositional surface pilot, not a complete grammar.
"""

from __future__ import annotations

import random

from document_world import COLORS, NAMES, PLACES
from document_world_v9_typed_lexicon import _KIND
from document_proposition_surface import (
    AnswerSite, RenderedDocument, answer_from_visible_prefix,
    parse_visible_document, realize,
)


def _attribute_for_value(value):
    kind = _KIND.get(value)
    if value in NAMES or kind == 'people':
        return 'owner'
    if value in COLORS or kind in ('colors_c', 'colors_v'):
        return 'color'
    if value in PLACES or kind in ('places_in', 'places_on'):
        return 'place'
    return None


def _prior_sentence_prefix(tokens, position):
    start = position
    while start and tokens[start-1] != '.':
        start -= 1
    return tuple(tokens[:start]), start


def _visible_attribute(prefix, obj, attr, lexical_enabled):
    query = ('At', 'this', 'point', ',', 'the', attr,
             'of', 'the', obj, 'was')
    return answer_from_visible_prefix(prefix + query, lexical_enabled)


def _relative_tokens(attr, value):
    if attr == 'owner':
        return (',', 'which', value, 'owns', ',')
    if attr == 'color':
        return (',', 'which', 'is', value, ',')
    assert attr == 'place'
    kind = _KIND.get(value)
    prep = ('on' if kind == 'places_on' or value in ('shelf', 'desk')
            else 'in')
    return (',', 'which', 'is', prep, 'the', value, ',')


def add_relative_clauses(doc: RenderedDocument, seed=0, rate=.75,
                         lexical_enabled=False):
    """Return (document, insertion count) with every site remapped."""
    if not 0 <= rate <= 1:
        raise ValueError(rate)
    tokens = doc.tokens
    if any(tokens[i:i+2] == (',', 'which') for i in range(len(tokens)-1)):
        raise ValueError('Reserved relative-clause marker already present')
    rng = random.Random(seed)
    inserts = {}
    for site in doc.sites:
        if site.family not in ('fact', 'update') or site.entity is None:
            continue
        target_attr = _attribute_for_value(site.answer)
        if target_attr is None or rng.random() >= rate:
            continue
        prior, start = _prior_sentence_prefix(tokens, site.index)
        entity_positions = [i for i in range(start, site.index)
                            if tokens[i] == site.entity]
        if len(entity_positions) != 1:
            continue
        obj_index = entity_positions[0]
        available = ('owner', 'color', 'place')
        other = rng.choice([attr for attr in available if attr != target_attr])
        value = _visible_attribute(prior, site.entity, other, lexical_enabled)
        if obj_index in inserts:
            continue
        inserts[obj_index] = _relative_tokens(other, value)
    result, indices = [], {}
    for i, token in enumerate(tokens):
        indices[i] = len(result)
        result.append(token)
        if i in inserts:
            result.extend(inserts[i])
    sites = tuple(AnswerSite(indices[site.index], site.family,
                             site.answer, site.candidates, site.entity)
                  for site in doc.sites)
    changed = RenderedDocument(tuple(result), sites, doc.form, doc.plan)
    for site in sites:
        assert changed.tokens[site.index] == site.answer
    return changed, len(inserts)


def _relative_claim(tokens, index):
    """Return (attribute, value, end_after_comma) for a visible clause."""
    assert tokens[index:index+2] == (',', 'which')
    if (index+5 <= len(tokens) and tokens[index+3:index+5] ==
            ('owns', ',')):
        return 'owner', tokens[index+2], index+5
    if (index+5 <= len(tokens) and tokens[index+2] == 'is' and
            tokens[index+4] == ','):
        return 'color', tokens[index+3], index+5
    if (index+7 <= len(tokens) and tokens[index+2] == 'is' and
            tokens[index+3] in ('in', 'on') and tokens[index+4] == 'the'
            and tokens[index+6] == ','):
        return 'place', tokens[index+5], index+7
    raise ValueError('Malformed relative clause')


def expand_checked(tokens, lexical_enabled=False):
    """Erase each relative clause only after checking its visible claim."""
    output = []
    i = 0
    while i < len(tokens):
        if tokens[i:i+2] != (',', 'which'):
            output.append(tokens[i])
            i += 1
            continue
        if not output:
            raise ValueError('Relative clause has no referent')
        obj = output[-1]
        attr, value, end = _relative_claim(tokens, i)
        if _attribute_for_value(value) != attr:
            raise AssertionError(('Wrong relative-clause value type', attr, value))
        if attr == 'place' and tokens[i+3] != _relative_tokens(attr, value)[3]:
            raise AssertionError(('Wrong location preposition', value))
        prior, _ = _prior_sentence_prefix(output, len(output))
        expected = _visible_attribute(prior, obj, attr, lexical_enabled)
        if value != expected:
            raise AssertionError((obj, attr, value, expected))
        i = end
    return tuple(output)


def parse_relative(doc: RenderedDocument, lexical_enabled=False):
    return parse_visible_document(expand_checked(doc.tokens, lexical_enabled),
                                  lexical_enabled)


def answer_from_relative_prefix(prefix, lexical_enabled=False):
    return answer_from_visible_prefix(expand_checked(prefix, lexical_enabled),
                                      lexical_enabled)


def realize_relative(plan, seed=0, relative_rate=.75,
                     lexical_exponent=None):
    base = realize(plan, seed=seed, lexical_exponent=lexical_exponent,
                   surface_version=3)
    return add_relative_clauses(
        base, seed=seed ^ 0x4E1A71B3, rate=relative_rate,
        lexical_enabled=lexical_exponent is not None)

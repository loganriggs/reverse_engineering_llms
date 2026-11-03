"""Compose an update with a separate true fact from the current world.

The side proposition is sampled from the pre-update state. Four clause
compositions put it before or after the update; the visible-only checker
checks its truth before recovering the scored answer. This is a narrow test
of compositional sentence realization, not a full document planner.
"""

from __future__ import annotations

import random

from document_world import VALUES, replay
from document_world_v5_clause_realizations import _sentences
from document_world_v8_discourse import (
    _extended_event, answer_from_visible_prefix as answer_v8,
    draw_varied as draw_v8, realize as realize_v8,
)
from document_world_v9_typed_lexicon import (
    _CANONICAL_POOLS, _KIND, draw_lexicon, lexicalize_document,
)

_SALT = 0xC0A105E


def plan(world, compose_rate=.75):
    assert 0 <= compose_rate <= 1
    rng = random.Random(world['realization_seed'] ^ _SALT)
    objects = tuple(world['initial'])
    return [dict(compose=rng.random() < compose_rate,
                 other=rng.choice([o for o in objects
                                   if o != event['object']]),
                 attribute=rng.choice(tuple(VALUES)),
                 position=rng.randrange(4),
                 event_form=rng.randrange(8))
            for event in world['events']]


def event_sentences(world):
    current = replay(world['initial'], [])
    sentences = {}
    for j, (event, source, use_alias, item) in enumerate(zip(
            world['events'], world['sources'], world['use_alias'],
            world['composed_event_plan'], strict=True)):
        ref = (world['aliases'][event['object']] if use_alias
               else event['object'])
        base = _extended_event(event, current, source, ref,
                               item['event_form'])
        if not item['compose']:
            sentences[j] = base
        else:
            other, attr = item['other'], item['attribute']
            value = current[other][attr]
            if item['position'] == 0:
                sentences[j] = (['While', 'the', other, 'had', value,
                                 'as', 'its', attr, ','] + base)
            elif item['position'] == 1:
                sentences[j] = (base[:-1] + [',', 'while', 'the', other,
                                'had', value, 'as', 'its', attr, '.'])
            elif item['position'] == 2:
                sentences[j] = (['With', other, 'listed', 'as', value,
                                 'for', attr, ','] + base)
            else:
                sentences[j] = (base[:-1] + [';', 'meanwhile', other,
                                'had', value, 'as', 'its', attr, '.'])
        current[event['object']][event['attribute']] = event['value']
    return sentences


def realize(world, lexical_enabled=False, **kwargs):
    doc = realize_v8(world, event_sentence_plan=event_sentences(world),
                     **kwargs)
    return lexicalize_document(doc, world) if lexical_enabled else doc


def draw_varied(rng, compose_rate=.75, lexical_exponent=None, **kwargs):
    base = draw_v8(rng, **kwargs)
    world = base.latent
    world['composed_event_plan'] = plan(world, compose_rate)
    if lexical_exponent is not None:
        world['surface_lexicon'] = draw_lexicon(
            random.Random(world['realization_seed'] ^ 0xC019),
            lexical_exponent)
    render_keys = ('syntax_entropy', 'aside_rate', 'context_rate',
                   'paragraph_rate', 'header_rate', 'intro_interleave_rate')
    return realize(world, lexical_enabled=lexical_exponent is not None,
                   **{key: kwargs[key] for key in render_keys
                      if key in kwargs})


def _query(other, attr):
    if attr == 'owner':
        return ['By', 'then', ',', 'the', other, 'belonged', 'to']
    if attr == 'color':
        return ['By', 'then', ',', 'the', other, 'was']
    return ['Current', 'location', 'of', other, ':']


def _uncompose(row):
    """Return (base sentence without period, other, attribute, value)."""
    if row[:2] == ['While', 'the']:
        assert row[3] == 'had' and row[5:7] == ['as', 'its']
        assert row[8] == ','
        return row[9:], row[2], row[7], row[4]
    if row[:1] == ['With']:
        assert row[2:4] == ['listed', 'as'] and row[5] == 'for'
        assert row[7] == ','
        return row[8:], row[1], row[6], row[4]
    for marker, lead, size in ((',', 'while', 9),
                               (';', 'meanwhile', 8)):
        idx = len(row) - size
        if idx >= 0 and row[idx:idx+2] == [marker, lead]:
            if marker == ',':
                assert row[idx+2] == 'the'
                other, start = row[idx+3], idx+4
            else:
                other, start = row[idx+2], idx+3
            assert row[start] == 'had'
            assert row[start+2:start+4] == ['as', 'its']
            assert start+5 == len(row)
            return row[:idx], other, row[start+4], row[start+1]
    return None


def _canonical_answer(prefix):
    complete, raw = _sentences([w for w in prefix if w != '\n'])
    retained = []
    for row in complete:
        parsed = _uncompose(row)
        if parsed is None:
            retained.extend(row + ['.'])
            continue
        base, other, attr, value = parsed
        assert attr in VALUES and value in VALUES[attr]
        observed = answer_v8(retained + _query(other, attr))
        assert observed == value, (other, attr, observed, value)
        retained.extend(base + ['.'])
    return answer_v8(retained + raw)


def answer_from_visible_prefix(prefix, lexical_enabled=False):
    if not lexical_enabled:
        return _canonical_answer(prefix)
    encountered = {kind: {} for kind in _CANONICAL_POOLS}
    reverse, normalized = {}, []
    for token in prefix:
        possessive = token.endswith("'s")
        base = token[:-2] if possessive else token
        kind = _KIND.get(base)
        if kind is None:
            normalized.append(token)
            continue
        assigned = encountered[kind]
        if base not in assigned:
            pool = _CANONICAL_POOLS[kind]
            if len(assigned) >= len(pool):
                raise ValueError(f'Too many visible {kind} symbols')
            assigned[base] = pool[len(assigned)]
            reverse[assigned[base]] = base
        canonical = assigned[base]
        normalized.append(canonical + "'s" if possessive else canonical)
    answer = _canonical_answer(normalized)
    if answer not in reverse:
        raise ValueError('Answer was not observed in visible prefix')
    return reverse[answer]

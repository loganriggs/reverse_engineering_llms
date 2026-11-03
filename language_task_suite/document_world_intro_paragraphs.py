"""Sample and compose initial world facts into variable paragraph clauses.

The V8 event world and scored sites stay fixed. Introductory facts are
realized as clauses that may share a sentence, with different grammatical
roles for owner, object, alias, and location. A visible-only transducer
recovers those facts before the existing V8 answer parser runs.
"""

from __future__ import annotations

import random

from document_world import NAMES, VALUES, _preposition
from document_world_v3_multisite import ALIASES, article
from document_world_v5_clause_realizations import _intro, _sentences
from document_world_v8_discourse import (
    answer_from_visible_prefix as answer_v8,
    draw_varied as draw_v8, realize as realize_v8,
)
from document_world_v9_typed_lexicon import (
    _CANONICAL_POOLS, _KIND, draw_lexicon, lexicalize_document,
)

_SALT = 0x1A770


def _clause(obj, state, alias, form):
    owner, color, place = (state[key] for key in ('owner', 'color', 'place'))
    prep = _preposition(place)
    if form == 0:
        return [owner, 'kept', article(color), color, obj, prep, 'the',
                place, ',', 'calling', 'it', alias]
    if form == 1:
        return ['The', color, obj, prep, 'the', place, 'belonged', 'to',
                owner, 'and', 'was', 'called', alias]
    if form == 2:
        return ['At', 'the', place, ',', owner, 'had', article(color),
                color, obj, 'under', 'the', 'name', alias]
    if form == 3:
        return [alias, 'was', 'the', 'name', 'of', owner + "'s", color,
                obj, prep, 'the', place]
    assert form == 4
    return ['For', owner, ',', 'the', color, obj, 'coded', alias,
            'sat', prep, 'the', place]


def plan(world, pairing_rate=.65):
    assert 0 <= pairing_rate <= 1
    rng = random.Random(world['realization_seed'] ^ _SALT)
    objects = list(world['initial'])
    rng.shuffle(objects)
    clauses = [_clause(obj, world['initial'][obj], world['aliases'][obj],
                       rng.randrange(5)) for obj in objects]
    sentences = []
    i = 0
    while i < len(clauses):
        size = 1 + int(i+1 < len(clauses) and rng.random() < pairing_rate)
        if size == 2 and i+2 < len(clauses) and rng.random() < .2:
            size = 3
        group = clauses[i:i+size]
        sentence = list(group[0])
        for clause in group[1:]:
            sentence.extend(rng.choice(([';'], [',', 'while'])) + clause)
        sentence.append('.')
        sentences.append(sentence)
        i += size
    return sentences


def realize(world, lexical_enabled=False, **kwargs):
    doc = realize_v8(world, initial_sentence_plan=plan(world), **kwargs)
    return lexicalize_document(doc, world) if lexical_enabled else doc


def draw_varied(rng, lexical_exponent=None, **kwargs):
    base = draw_v8(rng, **kwargs)
    world = base.latent
    if lexical_exponent is not None:
        world['surface_lexicon'] = draw_lexicon(
            random.Random(world['realization_seed'] ^ 0xC019),
            lexical_exponent)
    render_keys = ('syntax_entropy', 'aside_rate', 'context_rate',
                   'paragraph_rate', 'header_rate')
    return realize(world, lexical_enabled=lexical_exponent is not None,
                   **{key: kwargs[key] for key in render_keys
                      if key in kwargs})


def _is_intro(row):
    return (('calling' in row and row[0] in NAMES) or
            ('called' in row and row[0] == 'The') or
            (row[:2] == ['At', 'the'] and 'name' in row) or
            (row[0] in ALIASES and row[1:5] ==
             ['was', 'the', 'name', 'of']) or
            (row[0] == 'For' and 'coded' in row))


def _parse_clause(row):
    if row[0] in NAMES and row[1] == 'kept':
        (owner, _, det, color, obj, prep, the, place, comma,
         calling, it, alias) = row
        assert (det, prep, the, comma, calling, it) == (
            article(color), _preposition(place), 'the', ',',
            'calling', 'it')
    elif row[0] == 'The':
        (_, color, obj, prep, the, place, belonged, to, owner,
         and_, was, called, alias) = row
        assert (prep, the, belonged, to, and_, was, called) == (
            _preposition(place), 'the', 'belonged', 'to', 'and',
            'was', 'called')
    elif row[:2] == ['At', 'the']:
        (_, _, place, comma, owner, had, det, color, obj,
         under, the, name, alias) = row
        assert (comma, had, det, under, the, name) == (
            ',', 'had', article(color), 'under', 'the', 'name')
    elif row[0] in ALIASES:
        (alias, was, the, name, of, owner_s, color, obj,
         prep, the2, place) = row
        assert (was, the, name, of, prep, the2) == (
            'was', 'the', 'name', 'of', _preposition(place), 'the')
        assert owner_s.endswith("'s")
        owner = owner_s[:-2]
    else:
        assert row[0] == 'For'
        (_, owner, comma, the, color, obj, coded, alias,
         sat, prep, the2, place) = row
        assert (comma, the, coded, sat, prep, the2) == (
            ',', 'the', 'coded', 'sat', _preposition(place), 'the')
    assert owner in VALUES['owner'] and color in VALUES['color']
    assert place in VALUES['place'] and alias in ALIASES
    return _intro(obj, dict(owner=owner, color=color, place=place),
                  alias, 0)


def _canonical_answer(prefix):
    complete, raw = _sentences([word for word in prefix if word != '\n'])
    retained = []
    for row in complete:
        if not _is_intro(row):
            retained.extend(row + ['.'])
            continue
        pieces, fragment = [], []
        i = 0
        while i < len(row):
            if row[i:i+2] == [',', 'while']:
                pieces.append(fragment)
                fragment = []
                i += 2
            elif row[i] == ';':
                pieces.append(fragment)
                fragment = []
                i += 1
            else:
                fragment.append(row[i])
                i += 1
        pieces.append(fragment)
        for piece in pieces:
            retained.extend(_parse_clause(piece))
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

"""Typed lexical realization of V8's sampled document-world rules.

The V8 event ledger and syntax stay intact. A document samples consistent
surface words for its people, objects, colors, and locations. The answer
checker reads only the visible prefix: it renames typed words to temporary
canonical symbols in order of appearance, runs the independent V8 parser,
then renames the answer back. The temporary names need not match the hidden
world's original names.

This is a lexical DGP axis, not a natural-language realism claim or a
full-next-token marginal oracle.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from document_world import COLORS, NAMES, OBJECTS, PLACES
from document_world_v3_multisite import ALIASES, AnswerSite
from document_world_v5_clause_realizations import VariedDocument
from document_world_v8_discourse import (
    answer_from_visible_prefix as canonical_answer,
    draw_varied as draw_v8,
    realize as realize_v8,
)
from surface_mixture_v2 import (
    ADJECTIVES as INCIDENTAL_ADJECTIVES,
    EVENTS as INCIDENTAL_EVENTS,
    OBJECTS as INCIDENTAL_OBJECTS,
)


_ROWS = json.loads((Path(__file__).with_name('typed_lexicon_v17.json'))
                   .read_text())['categories']

_RESERVED = set('report record note item source color owner location label '
                'place change account update notice entry file story summary '
                'answer latest current most recent earlier new from to in on '
                'the a an was came by said wrote according registered kept '
                'stored known as belonged transferred handed painted moved '
                'recolored placed relocated'.split())
_RESERVED.update(INCIDENTAL_OBJECTS)
_RESERVED.update(word.lower() for word in ALIASES)


def _words(category, forbidden=()):
    return tuple(row['word'] for row in _ROWS[category]
                 if row['word'].isalpha() and row['word'].lower()
                 not in forbidden)


PEOPLE = _words('people', {word.lower() for word in ALIASES})
OBJECT_WORDS = _words('objects', _RESERVED)
ON_WORDS = tuple(word for word in (
    'table', 'counter', 'bench', 'platform', 'stage', 'wall', 'floor',
    'roof', 'balcony', 'porch', 'bridge', 'road', 'street', 'screen',
    'board', 'tray', 'seat', 'track') if word not in OBJECT_WORDS)
IN_WORDS = _words('interior_places', set(OBJECT_WORDS) | _RESERVED)
IN_WORDS = tuple(word for word in IN_WORDS
                 if word not in INCIDENTAL_EVENTS)
COLOR_CONSONANT = tuple(word for word in (
                   'crimson', 'navy', 'teal', 'violet', 'gold', 'silver',
                   'black', 'white', 'brown', 'gray', 'pink', 'cyan',
                   'maroon', 'turquoise', 'scarlet', 'beige', 'tan',
                   'lavender', 'cobalt', 'coral')
                   if word not in INCIDENTAL_ADJECTIVES)
COLOR_VOWEL = ('amber', 'indigo', 'ochre', 'olive', 'ivory', 'aqua')

_SURFACE_POOLS = dict(
    people=PEOPLE, objects=OBJECT_WORDS, colors_c=COLOR_CONSONANT,
    colors_v=COLOR_VOWEL, places_in=IN_WORDS, places_on=ON_WORDS)
_CANONICAL_POOLS = dict(
    people=NAMES, objects=OBJECTS,
    colors_c=tuple(word for word in COLORS if word[0] not in 'aeiou'),
    colors_v=tuple(word for word in COLORS if word[0] in 'aeiou'),
    places_in=tuple(word for word in PLACES if word not in ('shelf', 'desk')),
    places_on=('shelf', 'desk'))
assert all(len(_SURFACE_POOLS[k]) >= len(_CANONICAL_POOLS[k])
           for k in _SURFACE_POOLS)
assert len(set(word for pool in _SURFACE_POOLS.values() for word in pool)) == sum(
    map(len, _SURFACE_POOLS.values()))

_KIND = {word: kind for kind, pool in _SURFACE_POOLS.items()
         for word in pool}


def _draw_without_replacement(rng, words, n, exponent, category=None):
    if category:
        counts = {row['word']: row['dev_pos_count']
                  for row in _ROWS[category]}
    else:
        counts = {}
    remaining = list(words)
    selected = []
    for _ in range(n):
        weights = [(counts.get(word, 0) + 5)**exponent
                   for word in remaining]
        index = rng.choices(range(len(remaining)), weights)[0]
        selected.append(remaining.pop(index))
    return selected


def draw_lexicon(rng, frequency_exponent=0.):
    if frequency_exponent < 0:
        raise ValueError('frequency_exponent must be nonnegative')
    selected = {
        'people': _draw_without_replacement(
            rng, PEOPLE, len(NAMES), frequency_exponent, 'people'),
        'objects': _draw_without_replacement(
            rng, OBJECT_WORDS, len(OBJECTS), frequency_exponent, 'objects'),
        'colors_c': _draw_without_replacement(
            rng, COLOR_CONSONANT, len(_CANONICAL_POOLS['colors_c']),
            frequency_exponent),
        'colors_v': _draw_without_replacement(
            rng, COLOR_VOWEL, len(_CANONICAL_POOLS['colors_v']),
            frequency_exponent),
        'places_in': _draw_without_replacement(
            rng, IN_WORDS, len(_CANONICAL_POOLS['places_in']),
            frequency_exponent, 'interior_places'),
        'places_on': _draw_without_replacement(
            rng, ON_WORDS, len(_CANONICAL_POOLS['places_on']),
            frequency_exponent),
    }
    return {canonical: visible
            for kind, words in selected.items()
            for canonical, visible in zip(_CANONICAL_POOLS[kind], words)}


def _surface(token, mapping):
    if token.endswith("'s") and token[:-2] in mapping:
        return mapping[token[:-2]] + "'s"
    return mapping.get(token, token)


def lexicalize_document(canonical, world):
    assert 'surface_lexicon' in world
    mapping = world['surface_lexicon']
    tokens = tuple(
        token if (token == 'note' and canonical.tokens[i-2:i] ==
                  ('A', 'short') and canonical.tokens[i+1:i+2] ==
                  ('about',)) else _surface(token, mapping)
        for i, token in enumerate(canonical.tokens))
    sites = tuple(AnswerSite(site.index, site.family,
                             mapping.get(site.answer, site.answer),
                             tuple(mapping.get(c, c)
                                   for c in site.candidates),
                             site.object)
                  for site in canonical.sites)
    return VariedDocument(tokens, sites, canonical.genre, world)


def realize(world, **render_kwargs):
    return lexicalize_document(realize_v8(world, **render_kwargs), world)


def draw_varied(rng, frequency_exponent=0., **kwargs):
    canonical = draw_v8(rng, **kwargs)
    world = canonical.latent
    world['surface_lexicon'] = draw_lexicon(rng, frequency_exponent)
    render_keys = ('syntax_entropy', 'aside_rate', 'context_rate',
                   'paragraph_rate', 'header_rate', 'intro_interleave_rate')
    return realize(world, **{key: kwargs[key] for key in render_keys
                             if key in kwargs})


def answer_from_visible_prefix(prefix):
    """Reconstruct a target from visible words alone; no latent lexicon read."""
    encountered = {kind: {} for kind in _SURFACE_POOLS}
    reverse = {}
    normalized = []
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
    answer = canonical_answer(normalized)
    if answer not in reverse:
        raise ValueError('Answer was not observed in the visible prefix')
    return reverse[answer]

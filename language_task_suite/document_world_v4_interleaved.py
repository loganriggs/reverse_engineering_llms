"""Interleaved semantic continuations in a sampled event document.

This is a structure pilot. It varies the placement and number of prediction
sites and uses declarative recaps, but still has a small lexicon and no exact
full-document next-token conditional oracle. No latent facts are hidden from
the visible-prefix answer parser.
"""

from __future__ import annotations

import random

from document_world import VALUES, _preposition, replay
from document_world_v3_multisite import (ALIASES, GENRES, AnswerSite,
                                         WorldDocument, _event, _initial,
                                         draw_document, parse_visible)


def draw_interleaved(rng: random.Random, genre=None):
    world = draw_document(rng, genre)
    objects = tuple(world['initial'])
    plan = []
    if rng.random() < .65:
        plan.append((-1, 'alias_identity', rng.choice(objects), None))
    for j, event in enumerate(world['events']):
        if rng.random() < .18:
            plan.append((j, 'world_' + rng.choice(tuple(VALUES)),
                         rng.choice(objects), None))
        if rng.random() < .11:
            plan.append((j, 'source_attribution', event['object'],
                         event['attribute']))
    if not plan:
        plan.append((len(world['events']) - 1, 'world_' +
                     rng.choice(tuple(VALUES)), rng.choice(objects), None))
    world['recap_plan'] = plan
    return world


def _recap(tokens, sites, family, obj, attr, state, aliases, sources):
    if family == 'alias_identity':
        alias = aliases[obj]
        lead = ['The', 'item', 'labeled', alias, 'was', 'the']
        answer, candidates = obj, tuple(state)
    elif family == 'source_attribution':
        lead = ['The', 'latest', 'report', 'about', 'the', attr,
                'of', 'the', obj, 'came', 'from']
        answer, candidates = sources[(obj, attr)], VALUES['owner']
    else:
        attr = family.removeprefix('world_')
        if attr == 'owner':
            lead = ['By', 'then', ',', 'the', obj, 'belonged', 'to']
        elif attr == 'color':
            lead = ['By', 'then', ',', 'the', obj, 'was']
        else:
            lead = ['By', 'then', ',', 'the', obj, 'was',
                    _preposition(state[obj]['place']), 'the']
        answer, candidates = state[obj][attr], VALUES[attr]
    tokens.extend(lead)
    index = len(tokens)
    tokens.extend((answer, '.'))
    sites.append(AnswerSite(index, family, answer, tuple(candidates), obj))


def realize(world):
    genre = world['genre']
    assert genre in GENRES
    tokens, sites = [], []
    current = replay(world['initial'], [])
    sources = {}
    plan = world['recap_plan']
    for obj, state in world['initial'].items():
        tokens.extend(_initial(obj, state, world['aliases'][obj],
                               world['topic'], genre))
    for after, family, obj, attr in plan:
        if after == -1:
            _recap(tokens, sites, family, obj, attr, current,
                   world['aliases'], sources)
    for j, (event, source, use_alias) in enumerate(zip(
            world['events'], world['sources'], world['use_alias'], strict=True)):
        ref = world['aliases'][event['object']] if use_alias else event['object']
        tokens.extend(_event(event, current, source, ref, genre))
        current[event['object']][event['attribute']] = event['value']
        sources[(event['object'], event['attribute'])] = source
        for after, family, obj, attr in plan:
            if after == j:
                _recap(tokens, sites, family, obj, attr, current,
                       world['aliases'], sources)
    return WorldDocument(tuple(tokens), tuple(sites), genre, world)


def _is_recap(sentence):
    return (sentence[:2] == ['By', 'then'] or
            sentence[:3] == ['The', 'latest', 'report'] or
            sentence[:3] == ['The', 'item', 'labeled'])


def answer_from_visible_prefix(prefix):
    """Recover a recap answer using only complete visible prior sentences."""
    complete, partial = [], []
    for token in prefix:
        partial.append(token)
        if token == '.':
            sentence, partial = partial[:-1], []
            if not _is_recap(sentence):
                complete.extend(sentence + ['.'])
    assert partial
    state, aliases, sources, _ = parse_visible(complete)
    if partial[:3] == ['The', 'item', 'labeled']:
        assert partial[4:6] == ['was', 'the']
        return aliases[partial[3]]
    if partial[:3] == ['The', 'latest', 'report']:
        assert partial[3:8] == ['about', 'the', partial[5], 'of', 'the']
        assert partial[9:11] == ['came', 'from']
        return sources[(partial[8], partial[5])]
    assert partial[:4] == ['By', 'then', ',', 'the']
    obj = partial[4]
    if partial[5:7] == ['belonged', 'to']:
        return state[obj]['owner']
    assert partial[5] == 'was'
    if len(partial) == 6:
        return state[obj]['color']
    assert partial[6:8] == [_preposition(state[obj]['place']), 'the']
    return state[obj]['place']


def sample_documents(n, seed, genres=GENRES):
    rng = random.Random(seed)
    return [realize(draw_interleaved(rng, rng.choice(genres)))
            for _ in range(n)]

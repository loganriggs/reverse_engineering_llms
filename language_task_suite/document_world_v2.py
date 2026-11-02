"""Longer, grammatical realizations of the v1 latent document-world task.

This is a separate version: v1 checkpoints must continue to use their exact
original generator. The oracle, splits, and intervention semantics are shared.
"""

from __future__ import annotations

import json
import random

from document_world import (ATTRS, GENRES, VALUES, WorldExample,
                            _preposition, composition_fold,
                            replay, world_oracle)


def article(word):
    return 'an' if word[0].lower() in 'aeiou' else 'a'


def draw_world(rng: random.Random):
    from document_world import draw_world as draw_v1
    world = draw_v1(rng)
    # A seed rather than realized wording makes edits preserve all nuisance text.
    world['style_seed'] = rng.randrange(2**32)
    world['topic'] = rng.choice(('inventory', 'travel', 'archive', 'exhibit'))
    return world


def _sentence(tokens):
    return tokens + ['.']


def _initial(obj, state, index, topic, genre, rng):
    owner, color, place = (state[key] for key in ATTRS)
    prep = _preposition(place)
    if genre == 1:
        return _sentence(['The', topic, 'entry', 'for', 'the', obj, 'listed',
                          owner, 'as', 'its', 'owner', ',', color, 'as', 'its',
                          'color', ',', 'and', 'the', place, 'as', 'its', 'place'])
    if genre == 2:
        return _sentence(['At', 'the', 'start', 'of', 'the', topic, 'report',
                          ',', 'the', color, obj, 'was', prep,
                          'the', place, ',', 'and', owner, 'was', 'its', 'owner'])
    lead = rng.choice((['At', 'the', 'start', 'of', 'the', topic, 'report', ','],
                       ['According', 'to', 'the', 'initial', topic, 'record', ','],
                       ['Before', 'any', 'changes', 'were', 'made', ',']))
    return _sentence(lead + [owner, 'kept', article(color), color, obj, prep,
                             'the', place])


def _event(event, previous, topic, genre, rng):
    obj, attribute, value = (event[key] for key in ('object', 'attribute', 'value'))
    old = previous[obj][attribute]
    owner = previous[obj]['owner']
    if genre == 1:
        action = {'owner': ['transferred', 'from', owner, 'to', value],
                  'color': ['repainted', 'from', old, 'to', value],
                  'place': ['moved', 'from', 'the', old, 'to', 'the', value]}[attribute]
        return _sentence(['The', 'next', topic, 'entry', 'said', 'that', 'the',
                          obj, 'was'] + action + [',', 'while', 'the', 'other',
                          'items', 'were', 'not', 'changed', 'in', 'that', 'entry'])
    if genre == 2:
        action = {'owner': [owner, 'handed', 'the', obj, 'to', value],
                  'color': [owner, 'painted', 'the', obj, value],
                  'place': [owner, 'moved', 'the', obj, 'to', 'the', value]}[attribute]
        return _sentence(['Later', ',', 'the', topic, 'report', 'said', 'that'] +
                         action + [',', 'and', 'the', 'other', 'items',
                                   'were', 'not', 'changed', 'in', 'that', 'entry'])
    lead = rng.choice((['Later', 'in', 'the', topic, 'report', ','],
                       ['In', 'the', 'next', 'part', 'of', 'the', 'record', ','],
                       ['After', 'that', 'entry', 'was', 'written', ',']))
    action = {'owner': [owner, 'handed', 'the', obj, 'to', value,
                        ',', 'changing', 'who', 'owned', 'it'],
              'color': [owner, 'painted', 'the', obj, value,
                        ',', 'changing', 'its', 'color', 'from', old],
              'place': [owner, 'moved', 'the', obj, 'from', 'the', old,
                        'to', 'the', value]}[attribute]
    return _sentence(lead + action)


def _query(obj, attribute, topic, genre):
    if genre == 1:
        return ['The', 'final', topic, 'entry', 'should', 'list', 'the',
                attribute, 'of', 'the', obj, 'as'] + (['the'] if attribute == 'place' else [])
    if genre == 2:
        return ['After', 'all', 'the', 'changes', 'in', 'the', topic,
                'report', ',', 'the', attribute, 'of', 'the', obj, 'was'] + (
                    ['the'] if attribute == 'place' else [])
    return {'owner': ['After', 'all', 'the', 'changes', ',', 'the',
                      'owner', 'of', 'the', obj, 'was'],
            'color': ['After', 'all', 'the', 'changes', ',', 'the',
                      'color', 'of', 'the', obj, 'was'],
            'place': ['After', 'all', 'the', 'changes', ',', 'the',
                      'place', 'of', 'the', obj, 'was', 'the']}[attribute]


def realize(world: dict, genre: int) -> WorldExample:
    assert genre in range(len(GENRES))
    rng = random.Random(world['style_seed'])
    topic = world['topic']
    prefix = []
    for index, (obj, state) in enumerate(world['initial'].items()):
        prefix.extend(_initial(obj, state, index, topic, genre, rng))
    current = replay(world['initial'], [])
    for event in world['events']:
        prefix.extend(_event(event, current, topic, genre, rng))
        current[event['object']][event['attribute']] = event['value']
    obj, attribute = world['query_object'], world['query_attribute']
    prefix.extend(_query(obj, attribute, topic, genre))
    return WorldExample(f'world_{attribute}', tuple(prefix),
                        current[obj][attribute], VALUES[attribute], genre, world)


def sample_world(n: int, seed: int, split='train', genres=(0, 1)):
    if split not in ('train', 'composition'):
        raise ValueError(split)
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        world = draw_world(rng)
        example = realize(world, rng.choice(genres))
        if (composition_fold(example) == 0) == (split == 'composition'):
            out.append(example)
    return out


def causal_pair(example: WorldExample):
    world = json.loads(json.dumps(example.latent))
    obj, attribute = world['query_object'], world['query_attribute']
    index = [i for i, event in enumerate(world['events'])
             if event['object'] == obj and event['attribute'] == attribute][-1]
    old = world_oracle(example)
    world['events'][index]['value'] = next(v for v in VALUES[attribute] if v != old)
    edited = realize(world, example.variant)
    assert edited.target != example.target
    return example, edited


def nuisance_pair(example: WorldExample):
    world = json.loads(json.dumps(example.latent))
    event = world['events'][-1]
    old = event['value']
    event['value'] = next(v for v in VALUES[event['attribute']]
                          if v != old and v != example.target)
    edited = realize(world, example.variant)
    assert edited.target == example.target
    return example, edited

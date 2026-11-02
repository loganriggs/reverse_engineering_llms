"""Document-level latent-world tasks with compositional and genre holdouts.

The semantic object is a state-transition record. Several realizations express
the same record; the answer is computed by replay, never by template position.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass

NAMES = ('Maya', 'Liam', 'Noah', 'Ava', 'Owen', 'Iris', 'Emma', 'Leo', 'Nina', 'Theo')
OBJECTS = ('map', 'key', 'book', 'cup', 'note', 'photo', 'badge', 'ticket')
COLORS = ('red', 'blue', 'green', 'yellow', 'purple', 'orange')
PLACES = ('box', 'drawer', 'shelf', 'cabinet', 'desk', 'bag')
VALUES = {'owner': NAMES, 'color': COLORS, 'place': PLACES}
ATTRS = tuple(VALUES)
GENRES = ('story', 'notes', 'summary')


def _preposition(place):
    return 'on' if place in ('shelf', 'desk') else 'in'


@dataclass(frozen=True)
class WorldExample:
    family: str
    prefix: tuple[str, ...]
    target: str
    candidates: tuple[str, ...]
    variant: int
    latent: dict

    def text(self):
        return ' '.join(self.prefix + (self.target,)).replace(' .', '.').replace(' ,', ',')


def replay(initial: dict, events: list[dict]) -> dict:
    state = {obj: dict(values) for obj, values in initial.items()}
    for event in events:
        state[event['object']][event['attribute']] = event['value']
    return state


def _new_value(rng, attribute, previous, exclude=()):
    choices = [x for x in VALUES[attribute] if x != previous and x not in exclude]
    return rng.choice(choices)


def draw_world(rng: random.Random):
    objects = rng.sample(OBJECTS, 3)
    initial = {obj: {attribute: rng.choice(VALUES[attribute]) for attribute in ATTRS}
               for obj in objects}
    query_object = rng.choice(objects)
    query_attribute = rng.choice(ATTRS)
    events = []
    current = replay(initial, events)
    # Guarantee at least one semantically relevant update.
    first = _new_value(rng, query_attribute, current[query_object][query_attribute])
    events.append(dict(object=query_object, attribute=query_attribute, value=first))
    current[query_object][query_attribute] = first
    for _ in range(rng.randint(2, 5)):
        obj = rng.choice(objects)
        attribute = rng.choice(ATTRS)
        value = _new_value(rng, attribute, current[obj][attribute])
        events.append(dict(object=obj, attribute=attribute, value=value))
        current[obj][attribute] = value
    # The last mention of the answer type belongs to another object. It has a
    # different value, so a simple recency rule is guaranteed wrong.
    other = rng.choice([obj for obj in objects if obj != query_object])
    value = _new_value(rng, query_attribute, current[other][query_attribute],
                       exclude=(current[query_object][query_attribute],))
    events.append(dict(object=other, attribute=query_attribute, value=value))
    return dict(initial=initial, events=events,
                query_object=query_object, query_attribute=query_attribute)


def _initial_sentence(obj, state, genre):
    owner, color, place = (state[k] for k in ('owner', 'color', 'place'))
    if genre == 0:
        return [owner, 'kept', 'a', color, obj, _preposition(place), 'the', place, '.']
    if genre == 1:
        return ['Item', ':', obj, ';', 'owner', ':', owner, ';',
                'color', ':', color, ';', 'place', ':', place, '.']
    return ['the', color, obj, 'was', _preposition(place), 'the', place, '.',
            owner, 'kept', 'the', obj, '.']


def _event_sentence(event, previous, genre):
    obj, attribute, value = (event[k] for k in ('object', 'attribute', 'value'))
    owner = previous[obj]['owner']
    if genre == 0:
        if attribute == 'owner':
            return [owner, 'handed', 'the', obj, 'to', value, '.']
        if attribute == 'color':
            return [owner, 'painted', 'the', obj, value, '.']
        return [owner, 'moved', 'the', obj, 'to', 'the', value, '.']
    if genre == 1:
        verb = {'owner': 'Transfer', 'color': 'Recolor', 'place': 'Move'}[attribute]
        return [verb, ':', obj, 'to', value, '.']
    if attribute == 'owner':
        return ['the', obj, 'belonged', 'to', value, '.']
    if attribute == 'color':
        return ['the', obj, 'was', 'painted', value, '.']
    return ['the', obj, 'was', 'moved', 'to', 'the', value, '.']


def _query(obj, attribute, genre):
    if genre == 0:
        return {
            'owner': ['At', 'the', 'end', ',', 'the', obj, 'belonged', 'to'],
            'color': ['At', 'the', 'end', ',', 'the', 'color', 'of', 'the', obj, 'was'],
            'place': ['At', 'the', 'end', ',', 'the', 'final', 'location',
                      'of', 'the', obj, 'was', 'the'],
        }[attribute]
    if genre == 1:
        return ['Final', attribute, 'of', obj, ':']
    return ['At', 'the', 'end', ',', attribute, 'of', obj, ':']


def realize(world: dict, genre: int) -> WorldExample:
    assert genre in range(len(GENRES))
    initial, events = world['initial'], world['events']
    prefix = []
    for obj, state in initial.items():
        prefix.extend(_initial_sentence(obj, state, genre))
    current = replay(initial, [])
    for event in events:
        prefix.extend(_event_sentence(event, current, genre))
        current[event['object']][event['attribute']] = event['value']
    obj, attribute = world['query_object'], world['query_attribute']
    prefix.extend(_query(obj, attribute, genre))
    answer = current[obj][attribute]
    return WorldExample(f'world_{attribute}', tuple(prefix), answer,
                        VALUES[attribute], genre, world)


def composition_fold(example: WorldExample) -> int:
    """Hold out 1/5 of final queried object/value combinations, independent of prose."""
    key = f'{example.family}|{example.latent["query_object"]}|{example.target}'
    return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=4).digest(), 'big') % 5


def sample_world(n: int, seed: int, split='train', genres=(0, 1)):
    if split not in ('train', 'composition'):
        raise ValueError(split)
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        world = draw_world(rng)
        genre = rng.choice(genres)
        example = realize(world, genre)
        held_out = composition_fold(example) == 0
        if held_out == (split == 'composition'):
            out.append(example)
    return out


def world_oracle(example: WorldExample) -> str:
    world = example.latent
    state = replay(world['initial'], world['events'])
    return state[world['query_object']][world['query_attribute']]


def causal_pair(example: WorldExample):
    """Change the last relevant event's value, keeping all other events fixed."""
    world = json.loads(json.dumps(example.latent))
    obj, attribute = world['query_object'], world['query_attribute']
    matching = [i for i,e in enumerate(world['events'])
                if e['object'] == obj and e['attribute'] == attribute]
    index = matching[-1]
    old_answer = world_oracle(example)
    world['events'][index]['value'] = next(v for v in VALUES[attribute]
                                           if v != old_answer)
    edited = realize(world, example.variant)
    assert edited.target != example.target
    return example, edited


def nuisance_pair(example: WorldExample):
    """Change the final other-object event without changing the queried answer."""
    world = json.loads(json.dumps(example.latent))
    event = world['events'][-1]
    assert event['object'] != world['query_object']
    current = event['value']
    event['value'] = next(v for v in VALUES[event['attribute']]
                          if v != current and v != example.target)
    edited = realize(world, example.variant)
    assert edited.target == example.target
    return example, edited

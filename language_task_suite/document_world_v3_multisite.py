"""Versioned multi-task documents built from one visible event ledger.

Each sampled document contains initial facts, object labels, sourced state
changes, and five answer sites. `answer_from_visible_prefix` reconstructs
every scored answer from text alone. This is a semantic DGP pilot; its
small lexicon and fixed query endings are not yet FineWeb-like.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from document_world import (COLORS, NAMES, OBJECTS, PLACES, VALUES,
                            _preposition, draw_world, replay)


ALIASES = ('Cedar', 'Willow', 'Harbor', 'Dawn', 'River', 'Orchard',
           'Meadow', 'Pine', 'Birch', 'Summit', 'Marble', 'Amber')
TOPICS = ('inventory', 'archive', 'exhibit', 'shipment')
GENRES = ('story', 'log', 'summary')
VERB_ATTR = {'handed': 'owner', 'painted': 'color', 'moved': 'place'}
ATTR_VERB = {value: key for key, value in VERB_ATTR.items()}


def article(word):
    return 'an' if word[0].lower() in 'aeiou' else 'a'


@dataclass(frozen=True)
class AnswerSite:
    index: int
    family: str
    answer: str
    candidates: tuple[str, ...]
    object: str


@dataclass(frozen=True)
class WorldDocument:
    tokens: tuple[str, ...]
    sites: tuple[AnswerSite, ...]
    genre: str
    latent: dict

    def text(self):
        return (' '.join(self.tokens).replace(' .', '.')
                .replace(' ,', ',').replace(' ?', '?')
                .replace(' ;', ';').replace(' :', ':'))


def draw_document(rng: random.Random, genre=None):
    world = draw_world(rng)
    objects = tuple(world['initial'])
    world['aliases'] = dict(zip(objects, rng.sample(ALIASES, len(objects))))
    world['topic'] = rng.choice(TOPICS)
    world['genre'] = genre or rng.choice(GENRES)
    world['alias_query_object'] = rng.choice(objects)
    current = replay(world['initial'], [])
    sources = []
    for event in world['events']:
        actor = current[event['object']]['owner']
        sources.append(rng.choice([name for name in NAMES if name != actor]))
        current[event['object']][event['attribute']] = event['value']
    world['sources'] = sources
    world['use_alias'] = [rng.random() < .5 for _ in sources]
    # Ensure aliases are part of ordinary event prose, not just definitions.
    world['use_alias'][0] = True
    return world


def _initial(obj, state, alias, topic, genre):
    owner, color, place = (state[k] for k in ('owner', 'color', 'place'))
    if genre == 'log':
        return (['Item', ':', obj, ';', 'owner', ':', owner, ';',
                 'color', ':', color, ';', 'place', ':', place, ';',
                 'label', ':', alias, '.'])
    if genre == 'summary':
        return (['In', 'the', topic, 'report', ',', 'the', color, obj,
                 'belonged', 'to', owner, 'and', 'was', _preposition(place),
                 'the', place, '.', 'Its', 'label', 'was', alias, '.'])
    return ([owner, 'kept', article(color), color, obj,
             _preposition(place), 'the', place, '.',
             'The', obj, 'was', 'labeled', alias, '.'])


def _action(event, previous, ref):
    obj, attr, value = (event[k] for k in ('object', 'attribute', 'value'))
    actor = previous[obj]['owner']
    if attr == 'owner':
        return [actor, 'handed', 'the', ref, 'to', value]
    if attr == 'color':
        return [actor, 'painted', 'the', ref, value]
    return [actor, 'moved', 'the', ref, 'to', 'the', value]


def _event(event, previous, source, ref, genre):
    action = _action(event, previous, ref)
    if genre == 'log':
        return ['Source', ':', source, ';', 'action', ':', action[1], ';',
                'actor', ':', action[0], ';', 'item', ':', ref, ';',
                'value', ':', event['value'], '.']
    if genre == 'summary':
        return ['Later', ',', 'according', 'to', source, ','] + action + ['.']
    return [source, 'reported', 'that'] + action + ['.']


def _append_query(tokens, sites, prefix, answer, family, candidates, obj):
    tokens.extend(prefix)
    index = len(tokens)
    tokens.extend((answer, '.'))
    sites.append(AnswerSite(index, family, answer, tuple(candidates), obj))


def realize(world):
    genre = world['genre']
    assert genre in GENRES
    tokens, sites = [], []
    for obj, state in world['initial'].items():
        tokens.extend(_initial(obj, state, world['aliases'][obj],
                               world['topic'], genre))
    current = replay(world['initial'], [])
    for event, source, use_alias in zip(world['events'], world['sources'],
                                        world['use_alias'], strict=True):
        ref = world['aliases'][event['object']] if use_alias else event['object']
        tokens.extend(_event(event, current, source, ref, genre))
        current[event['object']][event['attribute']] = event['value']
    obj = world['query_object']
    alias_obj = world['alias_query_object']
    alias = world['aliases'][alias_obj]
    _append_query(tokens, sites, ['Which', 'item', 'was', 'labeled', alias, '?'],
                  alias_obj, 'alias_identity', world['initial'], alias_obj)
    attr = world['query_attribute']
    matching = [i for i, event in enumerate(world['events'])
                if event['object'] == obj and event['attribute'] == attr]
    assert matching
    source = world['sources'][matching[-1]]
    _append_query(tokens, sites,
                  ['Who', 'reported', 'the', 'last', 'change', 'to',
                   'the', attr, 'of', 'the', obj, '?'],
                  source, 'source_attribution', NAMES, obj)
    for attr, question in (
            ('owner', ['At', 'the', 'end', ',', 'who', 'owned', 'the', obj, '?']),
            ('color', ['At', 'the', 'end', ',', 'what', 'color', 'was', 'the', obj, '?']),
            ('place', ['At', 'the', 'end', ',', 'where', 'was', 'the', obj, '?'])):
        _append_query(tokens, sites, question, current[obj][attr],
                      'world_'+attr, VALUES[attr], obj)
    return WorldDocument(tuple(tokens), tuple(sites), genre, world)


def sample_documents(n, seed, genres=GENRES):
    rng = random.Random(seed)
    return [realize(draw_document(rng, rng.choice(genres))) for _ in range(n)]


def _sentences(prefix):
    complete, sentence = [], []
    for token in prefix:
        sentence.append(token)
        if token == '.':
            complete.append(sentence[:-1])
            sentence = []
    return complete, sentence


def _parse_action(action):
    actor, verb, det, ref, *tail = action
    assert actor in NAMES and verb in VERB_ATTR and det == 'the'
    attr = VERB_ATTR[verb]
    if attr == 'owner':
        assert tail[:1] == ['to'] and tail[1] in NAMES
        return actor, ref, attr, tail[1]
    if attr == 'color':
        assert len(tail) == 1 and tail[0] in COLORS
        return actor, ref, attr, tail[0]
    assert tail[:2] == ['to', 'the'] and tail[2] in PLACES
    return actor, ref, attr, tail[2]


def parse_visible(prefix):
    """Independently reconstruct state and sources from completed sentences."""
    sentences, current_query = _sentences(prefix)
    state, aliases, sources = {}, {}, {}
    for sentence in sentences:
        if not sentence:
            continue
        first = sentence[0]
        if first in NAMES and sentence[1] == 'kept':
            owner, _, det, color, obj, prep, the, place = sentence
            assert det == article(color) and the == 'the'
            assert prep == _preposition(place)
            state[obj] = dict(owner=owner, color=color, place=place)
            continue
        if first == 'The' and sentence[2:4] == ['was', 'labeled']:
            aliases[sentence[4]] = sentence[1]
            continue
        if first == 'Item':
            assert len(sentence) == 19
            _, colon, obj, semi, owner_key, colon2, owner, semi2, color_key, colon3, color, semi3, place_key, colon4, place, semi4, label_key, colon5, alias = sentence
            assert (colon, semi, owner_key, colon2, semi2, color_key,
                    colon3, semi3, place_key, colon4, semi4, label_key,
                    colon5) == (':', ';', 'owner', ':', ';', 'color', ':',
                                ';', 'place', ':', ';', 'label', ':')
            state[obj] = dict(owner=owner, color=color, place=place)
            aliases[alias] = obj
            continue
        if first == 'In':
            assert sentence[:6] == ['In', 'the', sentence[2], 'report', ',', 'the']
            assert sentence[8:14] == ['belonged', 'to', sentence[10],
                                      'and', 'was', _preposition(sentence[15])]
            assert sentence[14] == 'the' and len(sentence) == 16
            state[sentence[7]] = dict(owner=sentence[10], color=sentence[6],
                                      place=sentence[15])
            continue
        if first == 'Its':
            assert sentence[:3] == ['Its', 'label', 'was']
            assert state
            aliases[sentence[3]] = next(reversed(state))
            continue
        if first == 'Source':
            assert sentence[1:3] == [':', sentence[2]]
            assert sentence[3:6] == [';', 'action', ':']
            assert sentence[7:10] == [';', 'actor', ':']
            assert sentence[11:14] == [';', 'item', ':']
            assert sentence[15:18] == [';', 'value', ':']
            source, verb, actor, ref, value = (sentence[i] for i in
                                               (2, 6, 10, 14, 18))
            attr = VERB_ATTR[verb]
        elif first == 'Later':
            assert sentence[:4] == ['Later', ',', 'according', 'to']
            source = sentence[4]
            assert sentence[5] == ','
            actor, ref, attr, value = _parse_action(sentence[6:])
        elif len(sentence) > 3 and sentence[1:3] == ['reported', 'that']:
            source = sentence[0]
            actor, ref, attr, value = _parse_action(sentence[3:])
        else:
            # Completed queries do not alter world state.
            assert first in ('Which', 'Who', 'At'), sentence
            continue
        obj = aliases.get(ref, ref)
        assert obj in state and source in NAMES
        assert actor == state[obj]['owner']
        state[obj][attr] = value
        sources[(obj, attr)] = source
    return state, aliases, sources, current_query


def answer_from_visible_prefix(prefix):
    state, aliases, sources, query = parse_visible(prefix)
    assert query[-1] == '?', query
    if query[0] == 'Which':
        assert query[:5] == ['Which', 'item', 'was', 'labeled', query[4]]
        return aliases[query[4]]
    if query[0] == 'Who':
        assert query[:9] == ['Who', 'reported', 'the', 'last', 'change',
                             'to', 'the', query[7], 'of']
        return sources[(query[10], query[7])]
    assert query[:4] == ['At', 'the', 'end', ',']
    obj = query[-2]
    attr = ('owner' if query[4] == 'who' else
            'color' if query[4] == 'what' else 'place')
    return state[obj][attr]

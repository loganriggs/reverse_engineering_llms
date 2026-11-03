"""Exact-prefix event language with two realizations of one event.

This is the first grammar-frame correctness fixture after V14, not the
accepted realistic DGP. Both frames draw the same joint event distribution.
Rule proposals for object and verb compete at ordinary sentence slots.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import random


PEOPLE = ('Mara', 'Leo', 'Nina', 'Omar', 'Tara', 'Zoe')
ADJECTIVES = ('blue', 'old', 'small', 'sealed', 'heavy', 'fragile')
OBJECTS = ('parcel', 'letter', 'crate', 'painting', 'device', 'sample',
           'notebook', 'statue')
VERBS = ('carried', 'delivered', 'moved', 'sent', 'brought', 'taken')
PLACES = ('gallery', 'office', 'museum', 'clinic', 'studio', 'warehouse')
FUNCTION = ('As', 'for', 'the', ',', 'it', 'to', 'The', 'was', 'by', '.')
VOCAB = tuple(dict.fromkeys(FUNCTION + PEOPLE + ADJECTIVES + OBJECTS +
                            VERBS + PLACES))
TOKEN2ID = {token: i for i, token in enumerate(VOCAB)}
ID2TOKEN = {i: token for token, i in TOKEN2ID.items()}


@dataclass(frozen=True)
class Event:
    adjective: str
    object: str
    verb: str
    place: str
    person: str


def sentence(event: Event, frame: str):
    if frame == 'topical':
        return ('As', 'for', 'the', event.adjective, event.object, ',',
                event.person, event.verb, 'it', 'to', 'the', event.place,
                '.')
    if frame == 'passive':
        return ('The', event.adjective, event.object, 'was', event.verb,
                'to', 'the', event.place, 'by', event.person, '.')
    raise ValueError(frame)


def parse_event(tokens):
    tokens = tuple(tokens)
    if len(tokens) == 13 and tokens[:3] == ('As', 'for', 'the') and \
            tokens[5] == ',' and tokens[8:11] == ('it', 'to', 'the') and \
            tokens[-1] == '.':
        event = Event(tokens[3], tokens[4], tokens[7], tokens[11],
                      tokens[6])
        frame = 'topical'
    elif len(tokens) == 11 and tokens[0] == 'The' and \
            tokens[3] == 'was' and tokens[5:7] == ('to', 'the') and \
            tokens[8] == 'by' and tokens[-1] == '.':
        event = Event(tokens[1], tokens[2], tokens[4], tokens[7],
                      tokens[9])
        frame = 'passive'
    else:
        raise ValueError('Not a complete event sentence')
    if (event.adjective not in ADJECTIVES or event.object not in OBJECTS
            or event.verb not in VERBS or event.place not in PLACES
            or event.person not in PEOPLE):
        raise ValueError('Invalid event field')
    assert sentence(event, frame) == tokens
    return event, frame


def split_prefix(tokens):
    history, current = [], []
    for token in tokens:
        current.append(token)
        if token == '.':
            history.append(parse_event(current)[0])
            current = []
    return history, current


def zipf_prior(words, exponent=.9):
    weights = [1/(index+1)**exponent for index in range(len(words))]
    total = sum(weights)
    return {word: weight/total for word, weight in zip(words, weights)}


def mixture(parts):
    """Normalize active proposal masses; shared outcomes sum once."""
    total = sum(weight for weight, _ in parts.values())
    out = defaultdict(float)
    for weight, distribution in parts.values():
        for token, probability in distribution.items():
            out[token] += weight*probability/total
    assert abs(sum(out.values())-1) < 1e-12
    return dict(out)


class EventFrameLanguage:
    def __init__(self, seed=0, topical_rate=.5, zipf_exponent=.9):
        if not 0 < topical_rate < 1 or zipf_exponent < 0:
            raise ValueError('Invalid frame or Zipf control')
        self.rng = random.Random(seed)
        self.topical_rate = topical_rate
        self.zipf_exponent = zipf_exponent
        self.vocab, self.token2id, self.id2token = (
            VOCAB, TOKEN2ID, ID2TOKEN)

    def _prior(self, words):
        return zipf_prior(words, self.zipf_exponent)

    def object_components(self, history, adjective):
        parts = {'lexicon': (.35, self._prior(OBJECTS)),
                 'adjective_bigram': (.35, {
                     OBJECTS[(ADJECTIVES.index(adjective)*3+1) %
                             len(OBJECTS)]: 1.})}
        previous = next((event for event in reversed(history)
                         if event.adjective == adjective), None)
        if previous:
            parts['object_induction'] = (.20, {previous.object: 1.})
        if history:
            index = PLACES.index(history[-1].place)
            parts['place_skip'] = (.10, {
                OBJECTS[(index*5+2) % len(OBJECTS)]: 1.})
        return parts

    def verb_components(self, history, adjective, object_name):
        parts = {'lexicon': (.25, self._prior(VERBS)),
                 'adjective_bigram': (.25, {
                     VERBS[(ADJECTIVES.index(adjective)*2+1) %
                           len(VERBS)]: 1.}),
                 'object_relation': (.25, {
                     VERBS[(OBJECTS.index(object_name)*3+2) %
                           len(VERBS)]: 1.})}
        previous = next((event for event in reversed(history)
                         if event.object == object_name), None)
        if previous:
            parts['verb_induction'] = (.15, {previous.verb: 1.})
        if history:
            index = PLACES.index(history[-1].place)
            parts['place_skip'] = (.10, {
                VERBS[(index*4+3) % len(VERBS)]: 1.})
        return parts

    def next_distribution(self, prefix):
        tokens = [ID2TOKEN[item] if isinstance(item, int) else item
                  for item in prefix]
        history, current = split_prefix(tokens)
        if not current:
            return {'As': self.topical_rate,
                    'The': 1-self.topical_rate}
        frame = ('topical' if current[0] == 'As' else
                 'passive' if current[0] == 'The' else None)
        if frame is None:
            raise ValueError('Unrecognized frame prefix')
        pos = len(current)
        fixed = ({1: 'for', 2: 'the', 5: ',', 8: 'it', 9: 'to',
                  10: 'the', 12: '.'} if frame == 'topical' else
                 {3: 'was', 5: 'to', 6: 'the', 8: 'by', 10: '.'})
        if pos in fixed:
            return {fixed[pos]: 1.}
        if pos == (3 if frame == 'topical' else 1):
            return self._prior(ADJECTIVES)
        if pos == (4 if frame == 'topical' else 2):
            adjective = current[3 if frame == 'topical' else 1]
            return mixture(self.object_components(history, adjective))
        if pos == (7 if frame == 'topical' else 4):
            adjective = current[3 if frame == 'topical' else 1]
            object_name = current[4 if frame == 'topical' else 2]
            return mixture(self.verb_components(history, adjective,
                                                object_name))
        if pos == (11 if frame == 'topical' else 7):
            return self._prior(PLACES)
        if pos == (6 if frame == 'topical' else 9):
            return self._prior(PEOPLE)
        raise ValueError(f'Invalid {frame} prefix length {pos}')

    def label(self, prefix):
        tokens = [ID2TOKEN[item] if isinstance(item, int) else item
                  for item in prefix]
        _, current = split_prefix(tokens)
        if not current:
            return 'frame_start'
        frame = 'topical' if current[0] == 'As' else 'passive'
        if len(current) == (4 if frame == 'topical' else 2):
            return 'mixed_object'
        if len(current) == (7 if frame == 'topical' else 4):
            return 'mixed_verb'
        return 'surface'

    def sample_sequence(self, length=256):
        tokens, labels, distributions = [], [], []
        for _ in range(length):
            q = self.next_distribution(tokens)
            token = self.rng.choices(tuple(q), weights=tuple(q.values()))[0]
            assert q[token] > 0
            labels.append(self.label(tokens))
            distributions.append(q)
            tokens.append(token)
        return [TOKEN2ID[token] for token in tokens], labels, distributions


def render(tokens):
    words = [ID2TOKEN[item] if isinstance(item, int) else item
             for item in tokens]
    text = ' '.join(words)
    return text.replace(' ,', ',').replace(' .', '.')

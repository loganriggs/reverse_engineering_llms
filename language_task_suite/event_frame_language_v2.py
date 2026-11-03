"""Five-frame event grammar with exact marginal next-token probabilities.

The same latent event law is used for every realization. In active and cleft
sentences the verb precedes its adjective/object cues, so the visible-prefix
oracle marginalizes those unobserved fields rather than reading a private
sampled event. This remains a small controlled grammar, not natural text.
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import random

from event_frame_language_v1 import Event, mixture, zipf_prior


PEOPLE = ('Mara', 'Leo', 'Nina', 'Omar', 'Tara', 'Zoe', 'Amira', 'Ben',
          'Clara', 'Daniel', 'Elena', 'Felix', 'Hannah', 'Isaac', 'Jules',
          'Lena', 'Marco', 'Nadia', 'Priya', 'Victor')
ADJECTIVES = ('blue', 'old', 'small', 'sealed', 'heavy', 'fragile',
              'damaged', 'rare', 'wooden', 'metal', 'painted', 'folded',
              'large', 'new', 'valuable', 'broken')
OBJECTS = ('parcel', 'letter', 'crate', 'painting', 'device', 'sample',
           'notebook', 'statue', 'book', 'instrument', 'package', 'document',
           'box', 'photograph', 'poster', 'journal', 'envelope', 'manuscript',
           'camera', 'bag', 'sculpture', 'report', 'model', 'tablet')
VERBS = ('carried', 'delivered', 'moved', 'sent', 'brought', 'took')
PASSIVE_VERBS = {'took': 'taken'}
PASSIVE_VERBS_REVERSE = {surface: lemma for lemma, surface in
                         PASSIVE_VERBS.items()}
PLACES = ('gallery', 'office', 'museum', 'clinic', 'studio', 'warehouse',
          'archive', 'lab', 'library', 'school', 'hall', 'workshop',
          'depot', 'center', 'apartment', 'station')

# Every frame describes the same transfer event. The initial visible token
# identifies its frame (proper names identify the active form).
TEMPLATES = {
    'active': ('$person', '$verb', 'the', '$adjective', '$object',
               'to', 'the', '$place', '.'),
    'topical': ('As', 'for', 'the', '$adjective', '$object', ',',
                '$person', '$verb', 'it', 'to', 'the', '$place', '.'),
    'passive': ('The', '$adjective', '$object', 'was', '$verb',
                'to', 'the', '$place', 'by', '$person', '.'),
    'it_cleft': ('It', 'was', '$person', 'who', '$verb', 'the',
                 '$adjective', '$object', 'to', 'the', '$place', '.'),
    'what_cleft': ('What', '$person', '$verb', 'to', 'the',
                   '$place', 'was', 'the', '$adjective', '$object', '.'),
}
FRAME_WEIGHTS = {'active': .45, 'topical': .12, 'passive': .22,
                 'it_cleft': .13, 'what_cleft': .08}
FUNCTION = ('the', 'to', 'As', 'for', ',', 'it', 'The', 'was', 'by',
            'It', 'who', 'What', '.', 'taken')
VOCAB = tuple(dict.fromkeys(FUNCTION + PEOPLE + ADJECTIVES + OBJECTS +
                            VERBS + PLACES))
TOKEN2ID = {token: i for i, token in enumerate(VOCAB)}
ID2TOKEN = {i: token for token, i in TOKEN2ID.items()}
FIELDS = {'adjective': ADJECTIVES, 'object': OBJECTS,
          'verb': VERBS, 'place': PLACES, 'person': PEOPLE}


def sentence(event: Event, frame: str):
    return tuple((PASSIVE_VERBS.get(event.verb, event.verb)
                  if frame == 'passive' and item == '$verb'
                  else getattr(event, item[1:]))
                 if item.startswith('$') else item
                 for item in TEMPLATES[frame])


def frame_from_first(token):
    if token in PEOPLE:
        return 'active'
    return {'As': 'topical', 'The': 'passive',
            'It': 'it_cleft', 'What': 'what_cleft'}[token]


def parse_event(tokens):
    tokens = tuple(tokens)
    frame = frame_from_first(tokens[0])
    template = TEMPLATES[frame]
    if len(tokens) != len(template):
        raise ValueError('Incomplete event sentence')
    fields = {}
    for actual, expected in zip(tokens, template):
        if expected.startswith('$'):
            name = expected[1:]
            if name == 'verb' and frame == 'passive':
                actual = PASSIVE_VERBS_REVERSE.get(actual, actual)
            if actual not in FIELDS[name]:
                raise ValueError(f'Invalid {name}: {actual}')
            fields[name] = actual
        elif actual != expected:
            raise ValueError(f'Expected {expected}, got {actual}')
    event = Event(**fields)
    assert sentence(event, frame) == tokens
    return event, frame


def split_prefix(tokens):
    history, current = [], []
    for token in tokens:
        current.append(token)
        if token == '.':
            history.append(parse_event(current)[0])
            current = []
    return tuple(history), current


class EventFrameLanguageV2:
    def __init__(self, seed=0, frame_weights=None, zipf_exponent=.9):
        self.rng = random.Random(seed)
        self.frame_weights = dict(FRAME_WEIGHTS if frame_weights is None
                                  else frame_weights)
        if (set(self.frame_weights) != set(TEMPLATES) or
                min(self.frame_weights.values()) <= 0 or
                abs(sum(self.frame_weights.values())-1) > 1e-10 or
                zipf_exponent < 0):
            raise ValueError('Invalid frame or Zipf controls')
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
        old = next((event for event in reversed(history)
                    if event.adjective == adjective), None)
        if old:
            parts['object_induction'] = (.20, {old.object: 1.})
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
        old = next((event for event in reversed(history)
                    if event.object == object_name), None)
        if old:
            parts['verb_induction'] = (.15, {old.verb: 1.})
        if history:
            index = PLACES.index(history[-1].place)
            parts['place_skip'] = (.10, {
                VERBS[(index*4+3) % len(VERBS)]: 1.})
        return parts

    @lru_cache(maxsize=64)
    def _semantic_table(self, history):
        """P(adjective, object, verb | completed visible events)."""
        table = []
        for adjective, pa in self._prior(ADJECTIVES).items():
            objects = mixture(self.object_components(history, adjective))
            for object_name, po in objects.items():
                verbs = mixture(self.verb_components(
                    history, adjective, object_name))
                for verb, pv in verbs.items():
                    table.append((adjective, object_name, verb, pa*po*pv))
        assert abs(sum(row[3] for row in table)-1) < 1e-10
        return tuple(table)

    def next_distribution(self, prefix):
        tokens = [ID2TOKEN[token] if isinstance(token, int) else token
                  for token in prefix]
        history, current = split_prefix(tokens)
        if not current:
            out = {TEMPLATES[frame][0]: mass for frame, mass in
                   self.frame_weights.items() if frame != 'active'}
            for person, mass in self._prior(PEOPLE).items():
                out[person] = self.frame_weights['active']*mass
            return out
        frame = frame_from_first(current[0])
        template = TEMPLATES[frame]
        pos = len(current)
        if pos >= len(template):
            raise ValueError('Overlong sentence prefix')
        observed = {}
        for actual, expected in zip(current, template):
            if expected.startswith('$'):
                name = expected[1:]
                observed[name] = (PASSIVE_VERBS_REVERSE.get(actual, actual)
                                  if name == 'verb' and frame == 'passive'
                                  else actual)
            elif actual != expected:
                raise ValueError(f'Invalid frame prefix: {actual}')
        next_slot = template[pos]
        if not next_slot.startswith('$'):
            return {next_slot: 1.}
        field = next_slot[1:]
        if field in ('person', 'place'):
            return self._prior(FIELDS[field])
        counts = defaultdict(float)
        for adjective, object_name, verb, mass in self._semantic_table(
                history):
            row = {'adjective': adjective, 'object': object_name,
                   'verb': verb}
            if any(row[name] != value for name, value in observed.items()
                   if name in row):
                continue
            counts[row[field]] += mass
        total = sum(counts.values())
        if total <= 0:
            raise ValueError('Impossible semantic prefix')
        if field == 'verb' and frame == 'passive':
            return {PASSIVE_VERBS.get(word, word): mass/total
                    for word, mass in counts.items()}
        return {word: mass/total for word, mass in counts.items()}

    def label(self, prefix):
        tokens = [ID2TOKEN[token] if isinstance(token, int) else token
                  for token in prefix]
        _, current = split_prefix(tokens)
        if not current:
            return 'frame_start'
        frame = frame_from_first(current[0])
        slot = TEMPLATES[frame][len(current)]
        return {'$object': 'mixed_object', '$verb': 'mixed_verb'}.get(
            slot, 'surface')

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
    return ' '.join(words).replace(' ,', ',').replace(' .', '.')

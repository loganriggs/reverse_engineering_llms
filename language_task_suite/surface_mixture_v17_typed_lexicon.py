"""V2 rule mixture with a larger, frequency-controlled typed lexicon.

The exact visible-prefix oracle is retained. This is a controlled lexical
intervention, not a claim that the resulting prose resembles web text.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

from surface_mixture_language import LEADS, PLACE_PREPS, TIME_PREPS, VERBS, uniform


def unique(*groups):
    return tuple(dict.fromkeys(item for group in groups for item in group))


_LEXICON = json.loads((Path(__file__).with_name('typed_lexicon_v17.json')).read_text())['categories']
PEOPLE = tuple(row['word'] for row in _LEXICON['people'])
OBJECTS = tuple(row['word'] for row in _LEXICON['objects'])
PLACES = tuple(row['word'] for row in _LEXICON['places'])
INTERIOR_PLACES = tuple(row['word'] for row in _LEXICON['interior_places'])
EVENTS = tuple(row['word'] for row in _LEXICON['events'])
ADJECTIVES = tuple(row['word'] for row in _LEXICON['adjectives'])
CUES = LEADS + ('after', 'before', 'At', 'When', 'In', 'During')
VOCAB = unique(LEADS, PEOPLE, VERBS, ADJECTIVES, OBJECTS, PLACES, EVENTS,
               PLACE_PREPS, TIME_PREPS,
               ('At', 'When', 'In', 'During', 'after', 'before', 'ended',
                'the', ',', '.', 'because', 'had', 'asked', 'about', 'it',
                'earlier', 'in', 'and', 'wanted', 'a', 'written', 'record',
                'while', 'watches', 'according', 'to',
                'although', 'said', 'was', 'missing'))
TOKEN2ID = {token: i for i, token in enumerate(VOCAB)}
ID2TOKEN = {i: token for token, i in TOKEN2ID.items()}
DEFAULT_WEIGHTS = dict(lexical=.40, induction=.25, lead=.20, place_skip=.15)
FRAME_WEIGHTS = dict(lead=.15, subject=.15, at=.15, when=.15,
                     in_place=.20, during=.20)


def _split(tokens):
    history, clause = [], []
    for token in tokens:
        clause.append(token)
        if token == '.':
            history.append(clause)
            clause = []
    return history, clause


def _frame(clause):
    first = clause[0]
    if first in LEADS:
        return 'lead'
    if first in PEOPLE:
        return 'subject'
    if first == 'At':
        return 'at'
    if first == 'When':
        return 'when'
    if first == 'In':
        return 'in_place'
    if first == 'During':
        return 'during'
    raise ValueError(f'Unknown sentence start {first}')


def _indices(frame):
    return {'lead': (2, 3, 9, 13),
            'subject': (0, 6, 12, 13),
            'at': (4, 5, 2, 12),
            'when': (5, 6, 12, 13),
            'in_place': (3, 4, 2, 11),
            'during': (3, 4, 10, 11)}[frame]


def _metadata(clause):
    frame = _frame(clause)
    subject_i, verb_i, place_i, _ = _indices(frame)
    cue = (clause[0] if frame == 'lead' else
           clause[2] if frame == 'subject' else
           clause[0])
    return dict(frame=frame, subject=clause[subject_i],
                verb=clause[verb_i], place=clause[place_i], cue=cue)


def _tail(suffix, dist):
    """Next-token distribution after a frame's fixed base phrase."""
    if not suffix:
        return {'.': .35, ',': .30, 'because': .35}
    if suffix[0] == 'because':
        slots = (dist('people'), {'had': 1.0}, {'asked': 1.0},
                 {'about': 1.0}, {'it': 1.0}, {'earlier': 1.0})
        if len(suffix) <= 6:
            return slots[len(suffix)-1]
        if len(suffix) == 7:
            return {'.': .55, 'in': .45}
        assert suffix[7] == 'in'
        long_slots = ({'the': 1.0}, dist('places'), {'after': 1.0},
                      {'the': 1.0}, dist('events'), {'and': 1.0},
                      {'wanted': 1.0}, {'a': 1.0}, {'written': 1.0},
                      {'record': 1.0}, {'.': 1.0})
        return long_slots[len(suffix)-8]
    assert suffix[0] == ','
    if len(suffix) == 1:
        return {'while': .4, 'according': .3, 'although': .3}
    kind = suffix[1]
    if kind == 'according':
        slots = ({'to': 1.0}, dist('people'), {'.': 1.0})
        return slots[len(suffix)-2]
    if kind == 'although':
        slots = (dist('people'), {'said': 1.0}, {'the': 1.0},
                 dist('adjectives'), dist('objects'), {'was': 1.0},
                 {'missing': 1.0}, {'.': 1.0})
        return slots[len(suffix)-2]
    assert kind == 'while'
    slots = (dist('people'), {'watches': 1.0})
    if len(suffix) < 4:
        return slots[len(suffix)-2]
    if len(suffix) == 4:
        return {'.': .55, ',': .45}
    assert suffix[4] == ','
    if len(suffix) == 5:
        return {'according': .5, 'although': .5}
    second = suffix[5]
    if second == 'according':
        slots = ({'to': 1.0}, dist('people'), {'.': 1.0})
        return slots[len(suffix)-6]
    assert second == 'although'
    slots = (dist('people'), {'said': 1.0}, {'the': 1.0},
             dist('adjectives'), dist('objects'), {'was': 1.0},
             {'missing': 1.0}, {'.': 1.0})
    return slots[len(suffix)-6]


class SurfaceMixtureV17:
    def __init__(self, seed=0, weights=None, subject_repeat_mass=.4,
                 subject_return_mass=0., frequency_exponent=0.,
                 frequency_smoothing=5.):
        self.rng = random.Random(seed)
        self.weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
        assert set(self.weights) == set(DEFAULT_WEIGHTS)
        assert min(self.weights.values()) > 0
        if (subject_repeat_mass < 0 or subject_return_mass < 0 or
                subject_repeat_mass + subject_return_mass > 1):
            raise ValueError('Subject repeat and return masses must be '
                             'nonnegative and sum to at most 1')
        self.subject_repeat_mass = subject_repeat_mass
        self.subject_return_mass = subject_return_mass
        if frequency_exponent < 0 or frequency_smoothing <= 0:
            raise ValueError('Frequency exponent must be nonnegative and smoothing positive')
        self.frequency_exponent = frequency_exponent
        self.frequency_smoothing = frequency_smoothing
        self._distributions = {}
        for category, rows in _LEXICON.items():
            values = [(row['dev_pos_count'] + frequency_smoothing)**frequency_exponent
                      for row in rows]
            total = sum(values)
            self._distributions[category] = {
                row['word']: value/total for row,value in zip(rows, values)}
        self.vocab, self.token2id, self.id2token = VOCAB, TOKEN2ID, ID2TOKEN

    def _dist(self, category):
        return self._distributions[category]

    def _tokens(self, prefix):
        return [ID2TOKEN[i] if isinstance(i, int) else i for i in prefix]

    def _subject_distribution(self, history):
        if not history:
            return self._dist('people')
        previous = _metadata(history[-1])['subject']
        earlier = tuple(dict.fromkeys(_metadata(clause)['subject']
                                      for clause in history[:-1]
                                      if _metadata(clause)['subject'] != previous))
        out = {subject: (1-self.subject_repeat_mass-
                         self.subject_return_mass)*mass
               for subject,mass in self._dist('people').items()}
        out[previous] += self.subject_repeat_mass
        if earlier:
            for subject in earlier:
                out[subject] += self.subject_return_mass/len(earlier)
        else:
            # No non-immediate return target exists yet. Preserve full support
            # and normalized mass without silently raising local repetition.
            for subject,mass in self._dist('people').items():
                out[subject] += self.subject_return_mass*mass
        return out

    def candidates(self, prefix):
        history, clause = _split(self._tokens(prefix))
        frame = _frame(clause)
        subject_i, verb_i, _, _ = _indices(frame)
        if len(clause) != verb_i:
            raise ValueError('Candidates exist only at a verb slot')
        subject = clause[subject_i]
        cue = (clause[0] if frame != 'subject' else clause[2])
        subject_i = PEOPLE.index(subject)
        base = (subject_i*3) % len(VERBS)
        out = {'lexical': {VERBS[(base+j)%len(VERBS)]: p
                           for j, p in enumerate((.7, .2, .1))},
               'lead': {VERBS[(CUES.index(cue)*5+1)%len(VERBS)]: 1.0}}
        earlier = [_metadata(old) for old in history]
        matches = [old for old in earlier if old['subject'] == subject]
        if matches:
            out['induction'] = {matches[-1]['verb']: 1.0}
        if earlier:
            place = earlier[-1]['place']
            out['place_skip'] = {VERBS[(PLACES.index(place)*7+2)%len(VERBS)]: 1.0}
        return out

    def next_distribution(self, prefix):
        history, clause = _split(self._tokens(prefix))
        if not clause:
            subjects = self._subject_distribution(history)
            out = {lead: FRAME_WEIGHTS['lead']/len(LEADS) for lead in LEADS}
            for subject, mass in subjects.items():
                out[subject] = FRAME_WEIGHTS['subject']*mass
            out['At'] = FRAME_WEIGHTS['at']
            out['When'] = FRAME_WEIGHTS['when']
            out['In'] = FRAME_WEIGHTS['in_place']
            out['During'] = FRAME_WEIGHTS['during']
            return out
        frame = _frame(clause)
        pos = len(clause)
        subject_i, verb_i, place_i, base_end = _indices(frame)
        if pos == verb_i:
            candidates = self.candidates(prefix)
            z = sum(self.weights[name] for name in candidates)
            out = defaultdict(float)
            for name, dist in candidates.items():
                for token, mass in dist.items():
                    out[token] += self.weights[name]*mass/z
            return dict(out)
        if pos >= base_end:
            return _tail(clause[base_end:], self._dist)
        if pos == subject_i:
            return self._subject_distribution(history)
        if pos == place_i:
            if frame in ('lead', 'subject'):
                prep_i = 7 if frame == 'lead' else 10
                return self._dist('interior_places' if clause[prep_i] == 'inside'
                                  else 'places')
            return self._dist('places')
        slots = {
            'lead': {1: {',': 1.0}, 4: {'the': 1.0},
                     5: self._dist('adjectives'), 6: self._dist('objects'),
                     7: uniform(PLACE_PREPS), 8: {'the': 1.0},
                     10: uniform(TIME_PREPS), 11: {'the': 1.0},
                     12: self._dist('events')},
            'subject': {1: {',': 1.0}, 2: {'after': .6, 'before': .4},
                        3: {'the': 1.0}, 4: self._dist('events'),
                        5: {',': 1.0}, 7: {'the': 1.0},
                        8: self._dist('adjectives'), 9: self._dist('objects'),
                        10: uniform(PLACE_PREPS), 11: {'the': 1.0}},
            'at': {1: {'the': 1.0}, 3: {',': 1.0},
                   6: {'the': 1.0}, 7: self._dist('adjectives'),
                   8: self._dist('objects'), 9: {'before': 1.0},
                   10: {'the': 1.0}, 11: self._dist('events')},
            'when': {1: {'the': 1.0}, 2: self._dist('events'),
                     3: {'ended': 1.0}, 4: {',': 1.0},
                     7: {'the': 1.0}, 8: self._dist('adjectives'),
                     9: self._dist('objects'), 10: {'near': 1.0},
                     11: {'the': 1.0}},
            'in_place': {1: {'the': 1.0}, 5: {'the': 1.0},
                         6: self._dist('adjectives'), 7: self._dist('objects'),
                         8: {'after': 1.0}, 9: {'the': 1.0},
                         10: self._dist('events')},
            'during': {1: {'the': 1.0}, 2: self._dist('events'),
                       5: {'the': 1.0}, 6: self._dist('adjectives'),
                       7: self._dist('objects'), 8: {'near': 1.0},
                       9: {'the': 1.0}},
        }
        return slots[frame][pos]

    def sample_sequence(self, length=72):
        tokens, labels, qs = [], [], []
        for _ in range(length):
            q = self.next_distribution(tokens)
            _, clause = _split(tokens)
            label = ('mixed_verb' if clause and
                     len(clause) == _indices(_frame(clause))[1]
                     else 'frame')
            token = self.rng.choices(list(q), weights=list(q.values()))[0]
            labels.append(label)
            qs.append(q)
            tokens.append(token)
        return [TOKEN2ID[token] for token in tokens], labels, qs

    def sample_batch(self, n, length=72):
        rows = [self.sample_sequence(length) for _ in range(n)]
        return ([r[0] for r in rows], [r[1] for r in rows],
                [r[2] for r in rows])


def render(ids):
    text = ' '.join(ID2TOKEN[i] if isinstance(i, int) else i for i in ids)
    for punctuation in (',', '.'):
        text = text.replace(' '+punctuation, punctuation)
    return text

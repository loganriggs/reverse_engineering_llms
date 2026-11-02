"""Versioned multi-frame joint-rule language with an exact prefix oracle.

Six sentence frames carry the same lexical/induction/lead/place competition at
different positions. The frame is identifiable from visible tokens, so there
is no hidden template choice in the next-token oracle. Optional tail clauses
vary length and syntax. This remains a controlled toy, not web-text matching.
"""

import random
from collections import defaultdict

from surface_mixture_language import (ADJECTIVES as BASE_ADJECTIVES,
                                      EVENTS as BASE_EVENTS,
                                      INTERIOR_PLACES as BASE_INTERIOR,
                                      LEADS, OBJECTS as BASE_OBJECTS,
                                      PEOPLE as BASE_PEOPLE,
                                      PLACES as BASE_PLACES,
                                      PLACE_PREPS, TIME_PREPS, VERBS, uniform)


def unique(*groups):
    return tuple(dict.fromkeys(item for group in groups for item in group))


PEOPLE = unique(BASE_PEOPLE, ('Abigail', 'Aaron', 'Amira', 'Ben', 'Clara',
    'Daniel', 'Elena', 'Felix', 'Georgia', 'Hannah', 'Isaac', 'Jules', 'Lena',
    'Marco', 'Nadia', 'Oscar', 'Priya', 'Quinn', 'Rosa', 'Sam', 'Tara', 'Uma',
    'Victor', 'Wendy'))
OBJECTS = unique(BASE_OBJECTS, ('document', 'package', 'parcel', 'wallet',
    'notebook', 'tablet', 'laptop', 'umbrella', 'bicycle', 'suitcase',
    'photograph', 'sculpture', 'report', 'invoice', 'receipt', 'passport',
    'journal', 'newspaper', 'calendar', 'blanket', 'radio', 'lantern',
    'compass', 'medal', 'poster', 'microphone', 'headphones', 'painting',
    'drawing', 'statue', 'trophy', 'certificate', 'envelope', 'manuscript',
    'sample', 'specimen', 'container', 'device', 'instrument', 'record'))
PLACES = unique(BASE_PLACES, ('museum', 'gallery', 'clinic', 'hospital',
    'village', 'town', 'city', 'studio', 'warehouse', 'platform', 'airport',
    'campus', 'kitchen', 'hallway', 'classroom', 'theater', 'cafe',
    'restaurant', 'courthouse', 'balcony', 'basement', 'apartment', 'lobby',
    'garage', 'temple', 'mountain', 'valley', 'river', 'forest', 'desert',
    'coast'))
INTERIOR_PLACES = unique(BASE_INTERIOR, ('museum', 'gallery', 'clinic',
    'hospital', 'studio', 'warehouse', 'airport', 'kitchen', 'hallway',
    'classroom', 'theater', 'cafe', 'restaurant', 'courthouse', 'basement',
    'apartment', 'lobby', 'garage', 'temple'))
EVENTS = unique(BASE_EVENTS, ('conference', 'lecture', 'concert', 'dinner',
    'ceremony', 'campaign', 'election', 'opening', 'inspection', 'interview',
    'launch', 'delivery', 'rehearsal', 'repair', 'protest', 'contest',
    'auction', 'celebration', 'exhibition', 'presentation', 'discussion',
    'appointment', 'trip', 'tour', 'training', 'competition'))
ADJECTIVES = unique(BASE_ADJECTIVES, ('ancient', 'modern', 'ordinary',
    'fragile', 'damaged', 'folded', 'sealed', 'printed', 'paper', 'digital',
    'plastic', 'metal', 'glass', 'ceramic', 'leather', 'cotton', 'quiet',
    'noisy', 'narrow', 'wide', 'portable', 'local', 'official', 'private',
    'public', 'personal', 'original', 'temporary', 'recent', 'familiar',
    'unusual', 'simple', 'complex', 'fresh', 'faded', 'missing'))
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


def _choice_with_repeat(items, previous, repeat_mass=.4):
    out = {item: (1-repeat_mass)/len(items) for item in items}
    out[previous] += repeat_mass
    return out


def _tail(suffix):
    """Next-token distribution after a frame's fixed base phrase."""
    if not suffix:
        return {'.': .35, ',': .30, 'because': .35}
    if suffix[0] == 'because':
        slots = (uniform(PEOPLE), {'had': 1.0}, {'asked': 1.0},
                 {'about': 1.0}, {'it': 1.0}, {'earlier': 1.0})
        if len(suffix) <= 6:
            return slots[len(suffix)-1]
        if len(suffix) == 7:
            return {'.': .55, 'in': .45}
        assert suffix[7] == 'in'
        long_slots = ({'the': 1.0}, uniform(PLACES), {'after': 1.0},
                      {'the': 1.0}, uniform(EVENTS), {'and': 1.0},
                      {'wanted': 1.0}, {'a': 1.0}, {'written': 1.0},
                      {'record': 1.0}, {'.': 1.0})
        return long_slots[len(suffix)-8]
    assert suffix[0] == ','
    if len(suffix) == 1:
        return {'while': .4, 'according': .3, 'although': .3}
    kind = suffix[1]
    if kind == 'according':
        slots = ({'to': 1.0}, uniform(PEOPLE), {'.': 1.0})
        return slots[len(suffix)-2]
    if kind == 'although':
        slots = (uniform(PEOPLE), {'said': 1.0}, {'the': 1.0},
                 uniform(ADJECTIVES), uniform(OBJECTS), {'was': 1.0},
                 {'missing': 1.0}, {'.': 1.0})
        return slots[len(suffix)-2]
    assert kind == 'while'
    slots = (uniform(PEOPLE), {'watches': 1.0})
    if len(suffix) < 4:
        return slots[len(suffix)-2]
    if len(suffix) == 4:
        return {'.': .55, ',': .45}
    assert suffix[4] == ','
    if len(suffix) == 5:
        return {'according': .5, 'although': .5}
    second = suffix[5]
    if second == 'according':
        slots = ({'to': 1.0}, uniform(PEOPLE), {'.': 1.0})
        return slots[len(suffix)-6]
    assert second == 'although'
    slots = (uniform(PEOPLE), {'said': 1.0}, {'the': 1.0},
             uniform(ADJECTIVES), uniform(OBJECTS), {'was': 1.0},
             {'missing': 1.0}, {'.': 1.0})
    return slots[len(suffix)-6]


class SurfaceMixtureV2:
    def __init__(self, seed=0, weights=None, subject_repeat_mass=.4):
        self.rng = random.Random(seed)
        self.weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
        assert set(self.weights) == set(DEFAULT_WEIGHTS)
        assert min(self.weights.values()) > 0
        if not 0 <= subject_repeat_mass <= 1:
            raise ValueError('subject_repeat_mass must lie in [0, 1]')
        self.subject_repeat_mass = subject_repeat_mass
        self.vocab, self.token2id, self.id2token = VOCAB, TOKEN2ID, ID2TOKEN

    def _tokens(self, prefix):
        return [ID2TOKEN[i] if isinstance(i, int) else i for i in prefix]

    def _subject_distribution(self, history):
        return (_choice_with_repeat(PEOPLE, _metadata(history[-1])['subject'],
                                    self.subject_repeat_mass)
                if history else uniform(PEOPLE))

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
            return _tail(clause[base_end:])
        if pos == subject_i:
            return self._subject_distribution(history)
        if pos == place_i:
            if frame in ('lead', 'subject'):
                prep_i = 7 if frame == 'lead' else 10
                return uniform(INTERIOR_PLACES if clause[prep_i] == 'inside'
                               else PLACES)
            return uniform(PLACES)
        slots = {
            'lead': {1: {',': 1.0}, 4: {'the': 1.0},
                     5: uniform(ADJECTIVES), 6: uniform(OBJECTS),
                     7: uniform(PLACE_PREPS), 8: {'the': 1.0},
                     10: uniform(TIME_PREPS), 11: {'the': 1.0},
                     12: uniform(EVENTS)},
            'subject': {1: {',': 1.0}, 2: {'after': .6, 'before': .4},
                        3: {'the': 1.0}, 4: uniform(EVENTS),
                        5: {',': 1.0}, 7: {'the': 1.0},
                        8: uniform(ADJECTIVES), 9: uniform(OBJECTS),
                        10: uniform(PLACE_PREPS), 11: {'the': 1.0}},
            'at': {1: {'the': 1.0}, 3: {',': 1.0},
                   6: {'the': 1.0}, 7: uniform(ADJECTIVES),
                   8: uniform(OBJECTS), 9: {'before': 1.0},
                   10: {'the': 1.0}, 11: uniform(EVENTS)},
            'when': {1: {'the': 1.0}, 2: uniform(EVENTS),
                     3: {'ended': 1.0}, 4: {',': 1.0},
                     7: {'the': 1.0}, 8: uniform(ADJECTIVES),
                     9: uniform(OBJECTS), 10: {'near': 1.0},
                     11: {'the': 1.0}},
            'in_place': {1: {'the': 1.0}, 5: {'the': 1.0},
                         6: uniform(ADJECTIVES), 7: uniform(OBJECTS),
                         8: {'after': 1.0}, 9: {'the': 1.0},
                         10: uniform(EVENTS)},
            'during': {1: {'the': 1.0}, 2: uniform(EVENTS),
                       5: {'the': 1.0}, 6: uniform(ADJECTIVES),
                       7: uniform(OBJECTS), 8: {'near': 1.0},
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

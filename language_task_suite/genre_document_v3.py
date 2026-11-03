"""Hidden-genre joint document pilot with an exact Bayesian prefix oracle.

Genre changes the probability of *continuing* with claims, profiles, events,
numbered steps and arithmetic problems. Every genre supports every form, so
evidence accumulates across the document instead of a header revealing the
genre. This is a structural pilot, not a natural-language realism match.
"""

from __future__ import annotations

import random

from joint_document_v1 import (NAMES, OBJECTS, TOPICS, TOKEN2ID as BASE_IDS,
                               _normalize, _split, _zipf, site_label)
from joint_document_v2 import next_distribution as joint_distribution


GENRES = ('court', 'encyclopedia', 'procedure', 'math')
PRIORS = {'court': .25, 'encyclopedia': .25, 'procedure': .25, 'math': .25}
HEADERS = ('Document', 'Notes')
EXTRA_TOPICS = ('evidence', 'history', 'instructions', 'arithmetic')
TITLE_TOPICS = TOPICS + EXTRA_TOPICS
TOPIC_MAP = dict(zip(EXTRA_TOPICS,
                     ('courthouse', 'museum', 'workshop', 'classroom')))
STEP_ACTIONS = ('Check', 'Move', 'Keep', 'Find', 'Take', 'Use', 'Store', 'Carry')
DIGITS = tuple(str(i) for i in range(10))
VOCAB = tuple(dict.fromkeys(tuple(BASE_IDS) + HEADERS + EXTRA_TOPICS +
                            ('Step', 'Problem', '+', '=', '0') + STEP_ACTIONS))
TOKEN2ID = {word: i for i, word in enumerate(VOCAB)}

# All factors are positive. No one surface form proves the hidden genre.
START_FACTORS = {
    'court': dict(name=1.3, section=1.0, claim=4.5, question=2.0,
                  event=.8, step=.35, problem=.3),
    'encyclopedia': dict(name=2.2, section=2.2, claim=.85, question=.7,
                         event=1.3, step=.35, problem=.35),
    'procedure': dict(name=.8, section=1.5, claim=.45, question=.4,
                      event=1.0, step=5.5, problem=.5),
    'math': dict(name=.5, section=.8, claim=.35, question=.35,
                 event=.6, step=1.0, problem=8.0),
}
TITLE_WEIGHTS = {
    'court': {'Document': .55, 'Notes': .45},
    'encyclopedia': {'Document': .80, 'Notes': .20},
    'procedure': {'Document': .45, 'Notes': .55},
    'math': {'Document': .50, 'Notes': .50},
}
TOPIC_BOOST = dict(court='evidence', encyclopedia='history',
                   procedure='instructions', math='arithmetic')


def _title_distribution(genre):
    topic = TOPIC_BOOST[genre]
    out = {word: .8*p for word, p in _zipf(TITLE_TOPICS).items()}
    out[topic] += .2
    return out


def _base_prefix(prefix):
    """Remove only V3-only completed forms before calling the V2 oracle."""
    history, raw = _split(prefix)
    out = []
    for sentence in history:
        if sentence[0] in HEADERS:
            out.extend(('Report', ':', TOPIC_MAP.get(sentence[2], sentence[2]),
                        '.'))
        elif sentence[0] not in ('Step', 'Problem'):
            out.extend(sentence)
    if raw and raw[0] not in ('Step', 'Problem'):
        out.extend(raw)
    return out


def _step_distribution(history, raw):
    pos = len(raw)
    number = 1+sum(sentence[0] == 'Step' for sentence in history)
    assert number <= 99
    if pos == 1:
        return {str(number): 1.}
    slots = {2: {':': 1.}, 3: _zipf(STEP_ACTIONS),
             4: {'the': 1.}, 5: _zipf(OBJECTS), 6: {'.': 1.}}
    return slots[pos]


def _problem_distribution(raw):
    pos = len(raw)
    if pos == 9:
        return {str(int(raw[2])+int(raw[4])): 1.}
    slots = {1: {':': 1.}, 2: _zipf(DIGITS, .4), 3: {'+': 1.},
             4: _zipf(DIGITS, .4), 5: {'=': 1.}, 6: {'?': 1.},
             7: {'Answer': 1.}, 8: {':': 1.},
             10: {'.': 1.}}
    return slots[pos]


def _category(word):
    if word in NAMES:
        return 'name'
    if word in ('Section', 'Part'):
        return 'section'
    if word in ('According', 'As'):
        return 'claim'
    if word == 'Who':
        return 'question'
    return 'event'


def _shared_distribution(prefix, regime):
    if not prefix:
        return None, None
    history, raw = _split(prefix)
    if not history:
        assert raw[0] in HEADERS
        return history, raw
    if raw and raw[0] in ('Step', 'Problem'):
        return history, raw
    return history, raw, joint_distribution(_base_prefix(prefix), regime)


def conditional_distribution(prefix, genre, regime='task_rich', shared=None):
    """Next-token law for one fixed, unobserved document genre."""
    assert genre in GENRES and regime in ('reference', 'task_rich')
    if not prefix:
        return TITLE_WEIGHTS[genre]
    if shared is None:
        shared = _shared_distribution(prefix, regime)
    history, raw = shared[:2]
    if not history:
        pos = len(raw)
        slots = {1: {':': 1.}, 2: _title_distribution(genre),
                 3: {'.': 1.}}
        return slots[pos]
    if raw:
        if raw[0] == 'Step':
            return _step_distribution(history, raw)
        if raw[0] == 'Problem':
            return _problem_distribution(raw)
    base = shared[2]
    if raw:
        return base
    factors = START_FACTORS[genre]
    out = {word: p*factors[_category(word)] for word, p in base.items()}
    if sum(sentence[0] == 'Step' for sentence in history) < 99:
        out['Step'] = .04*factors['step']
    out['Problem'] = .04*factors['problem']
    return _normalize(out)


def site_label_v3(prefix):
    if not prefix:
        return 'title_kind'
    history, raw = _split(prefix)
    if not history:
        return 'title_topic' if len(raw) == 2 else 'other'
    if not raw:
        return 'genre_sentence_start'
    if raw[0] == 'Step':
        return 'step_number' if len(raw) == 1 else 'step_action' if len(raw) == 3 else 'other'
    if raw[0] == 'Problem':
        return 'arithmetic_answer' if len(raw) == 9 else 'other'
    return site_label(_base_prefix(prefix))


class PosteriorTrace:
    """One-document exact Bayes filter, retaining every genre's support."""

    def __init__(self, regime='task_rich'):
        assert regime in ('reference', 'task_rich')
        self.regime = regime
        self.posterior = dict(PRIORS)
        self.words = []

    def predict(self):
        shared = _shared_distribution(self.words, self.regime)
        qs = {genre: conditional_distribution(self.words, genre,
                                               self.regime, shared)
              for genre in GENRES}
        mixed = {}
        for genre, q in qs.items():
            for word, p in q.items():
                mixed[word] = mixed.get(word, 0.)+self.posterior[genre]*p
        return _normalize(mixed), qs

    def observe(self, word, qs):
        assert word in TOKEN2ID
        likelihood = {genre: self.posterior[genre]*q.get(word, 0.)
                      for genre, q in qs.items()}
        self.posterior = _normalize(likelihood)
        self.words.append(word)


def next_distribution(prefix, regime='task_rich'):
    trace = PosteriorTrace(regime)
    for word in prefix:
        _, qs = trace.predict()
        trace.observe(word, qs)
    return trace.predict()[0]


def sample_sequence(seed, length=256, regime='task_rich'):
    rng = random.Random(seed)
    genre = rng.choices(tuple(PRIORS), weights=tuple(PRIORS.values()))[0]
    trace = PosteriorTrace(regime)
    words, labels, distributions = [], [], []
    for _ in range(length):
        marginal, qs = trace.predict()
        q = qs[genre]
        label = site_label_v3(words)
        word = rng.choices(tuple(q), weights=tuple(q.values()))[0]
        assert marginal.get(word, 0.) > 0
        words.append(word)
        labels.append(label)
        distributions.append(marginal)
        trace.observe(word, qs)
    return words, labels, distributions, genre

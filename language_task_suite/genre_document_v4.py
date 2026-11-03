"""V3 document program with several surface forms for the same facts.

This is a *surface-form intervention*: canonical V3 facts and their exact
visible-prefix oracle remain available through an unambiguous transducer.
It tests whether paraphrases reduce frame concentration; it is not yet a
realistic web-text generator.
"""

from __future__ import annotations

import random

from genre_document_v3 import (GENRES, PRIORS, TOKEN2ID as V3_IDS,
                               conditional_distribution as v3_conditional,
                               site_label_v3)
from joint_document_v1 import NAMES, _normalize, _split


NEW_WORDS = ('Filed', 'by', 'Which', 'source', 'In', 'Following', 'works')
assert not (set(NEW_WORDS) & set(V3_IDS)), 'new form token collides with V3'
VOCAB = tuple(V3_IDS) + NEW_WORDS
TOKEN2ID = {word: i for i, word in enumerate(VOCAB)}
SPLIT = dict(According=('Filed', .5), Who=('Which', .5),
             At=('In', .5), After=('Following', .5))


def _canonical_sentence(sentence):
    if not sentence:
        return sentence
    first = sentence[0]
    if first == 'Filed':
        out = ['According']
        if len(sentence) >= 2:
            assert sentence[1] == 'by'
            out.append('to')
        out.extend(sentence[2:3])
        if len(sentence) >= 4:
            assert sentence[3] == ':'
            out.append(',')
        out.extend(sentence[4:])
        return tuple(out)
    if first == 'Which':
        if len(sentence) >= 2:
            assert sentence[1] == 'source'
        return tuple(['Who'] + list(sentence[2:]))
    if first == 'In':
        return ('At',) + tuple(sentence[1:])
    if first == 'Following':
        return ('After',) + tuple(sentence[1:])
    if first in NAMES and len(sentence) >= 2 and sentence[1] == 'works':
        if len(sentence) == 2:
            return (first,)
        assert sentence[2] == 'as'
        return (first, 'is') + tuple(sentence[3:])
    return tuple(sentence)


def to_v3(prefix):
    """Map every supported partial/completed V4 sentence to a V3 prefix."""
    history, raw = _split(prefix)
    out = []
    for sentence in history:
        out.extend(_canonical_sentence(sentence))
    out.extend(_canonical_sentence(raw))
    return out


def conditional_distribution(prefix, genre, regime='task_rich'):
    assert genre in GENRES and regime in ('reference', 'task_rich')
    canonical = to_v3(prefix)
    base = v3_conditional(canonical, genre, regime)
    history, raw = _split(prefix)
    if not raw and history:
        out = dict(base)
        for old, (new, proportion) in SPLIT.items():
            if old in out:
                out[new] = out[old]*proportion
                out[old] *= 1-proportion
        return _normalize(out)
    if not raw:
        return base
    if raw[0] == 'Filed':
        if len(raw) == 1:
            assert base == {'to': 1.}
            return {'by': 1.}
        if len(raw) == 3:
            assert base == {',': 1.}
            return {':': 1.}
        return base
    if raw[0] == 'Which':
        if len(raw) == 1:
            assert base == {'described': 1.}
            return {'source': 1.}
        return base
    if raw[0] in ('In', 'Following'):
        return base
    if raw[0] in NAMES:
        if len(raw) == 1 and 'is' in base:
            out = dict(base)
            out['works'] = out['is']*.5
            out['is'] *= .5
            return _normalize(out)
        if len(raw) == 2 and raw[1] == 'works':
            assert base == {'is': 1.}
            return {'as': 1.}
    return base


def site_label(prefix):
    return site_label_v3(to_v3(prefix))


class PosteriorTrace:
    def __init__(self, regime='task_rich'):
        assert regime in ('reference', 'task_rich')
        self.regime = regime
        self.words = []
        self.posterior = dict(PRIORS)

    def predict(self):
        qs = {genre: conditional_distribution(self.words, genre, self.regime)
              for genre in GENRES}
        mixed = {}
        for genre, q in qs.items():
            for word, p in q.items():
                mixed[word] = mixed.get(word, 0.)+self.posterior[genre]*p
        return _normalize(mixed), qs

    def observe(self, word, qs):
        assert word in TOKEN2ID
        scores = {genre: self.posterior[genre]*q.get(word, 0.)
                  for genre, q in qs.items()}
        self.posterior = _normalize(scores)
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
        label = site_label(words)
        word = rng.choices(tuple(q), weights=tuple(q.values()))[0]
        assert marginal.get(word, 0.) > 0
        words.append(word)
        labels.append(label)
        distributions.append(marginal)
        trace.observe(word, qs)
    return words, labels, distributions, genre

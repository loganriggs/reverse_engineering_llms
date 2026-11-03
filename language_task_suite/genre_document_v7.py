"""V4 task document with a small exact-PCFG background branch.

Unlike the V5 finite sentence bank or V6 n-gram control, each background
sentence is sampled by a grammar and its oracle marginalizes over parses.
This is an oracle/architecture pilot, not a realism milestone.
"""

from __future__ import annotations

import random

from genre_document_v4 import (GENRES, PRIORS, TOKEN2ID as TASK_IDS,
                               conditional_distribution as task_conditional,
                               site_label as task_label)
from joint_document_v1 import NAMES, _normalize, _split
from small_genre_grammar_v7 import GRAMMARS, STARTS, VOCAB as BG_VOCAB


TASK_STARTS = set(NAMES) | {
    'Document', 'Notes', 'Report', 'Guide', 'Section', 'Part',
    'According', 'As', 'Filed', 'Who', 'Which', 'At', 'In',
    'After', 'Following', 'Step', 'Problem'}
assert not (STARTS & TASK_STARTS)
BACKGROUND_RATE = {'task_rich': .20, 'reference': .70}
VOCAB = tuple(TASK_IDS) + tuple(sorted(BG_VOCAB-set(TASK_IDS)))
TOKEN2ID = {word: i for i, word in enumerate(VOCAB)}


def _task_prefix(prefix):
    history, raw = _split(prefix)
    out = []
    for sentence in history:
        if sentence[0] not in STARTS:
            out.extend(sentence)
    if raw and raw[0] not in STARTS:
        out.extend(raw)
    return out


def conditional_distribution(prefix, genre, regime='task_rich'):
    assert genre in GENRES and regime in BACKGROUND_RATE
    history, raw = _split(prefix)
    if raw and raw[0] in STARTS:
        return {word: p for word, p in GRAMMARS[genre]
                .next_distribution(raw).items() if word != '<eos>'}
    q_task = task_conditional(_task_prefix(prefix), genre, regime)
    if not history or raw:
        return q_task
    p_bg = BACKGROUND_RATE[regime]
    q_bg = GRAMMARS[genre].next_distribution(())
    assert not (set(q_bg) & set(q_task))
    return _normalize({**{w: (1-p_bg)*p for w, p in q_task.items()},
                       **{w: p_bg*p for w, p in q_bg.items()}})


def site_label(prefix):
    history, raw = _split(prefix)
    if raw and raw[0] in STARTS:
        return 'background_word'
    if history and not raw:
        return 'document_sentence_start'
    return task_label(_task_prefix(prefix))


class PosteriorTrace:
    def __init__(self, regime='task_rich'):
        assert regime in BACKGROUND_RATE
        self.regime = regime
        self.words = []
        self.posterior = dict(PRIORS)

    def predict(self):
        qs = {genre: conditional_distribution(self.words, genre, self.regime)
              for genre in GENRES}
        mixed = {}
        for genre, q in qs.items():
            for word, p in q.items():
                mixed[word] = mixed.get(word, 0.)+self.posterior.get(genre, 0.)*p
        return _normalize(mixed), qs

    def observe(self, word, qs):
        assert word in TOKEN2ID
        self.posterior = _normalize({genre: self.posterior.get(genre, 0.)*
                                     q.get(word, 0.) for genre, q in qs.items()})
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
        label = site_label(words)
        word = rng.choices(tuple(qs[genre]),
                           weights=tuple(qs[genre].values()))[0]
        assert marginal.get(word, 0.) > 0
        words.append(word)
        labels.append(label)
        distributions.append(marginal)
        trace.observe(word, qs)
    return words, labels, distributions, genre

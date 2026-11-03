"""Stochastic n-gram background control for the V5 joint task document.

V5 retrieves whole natural sentences, making its next-token oracle nearly
deterministic after a short prefix. V6 uses the same corpus-derived vocabulary
and sentence starts but samples later tokens from an interpolated 3/2/1-gram
model. The persistent common-word component prevents finite-bank prefix
memorization. This is a local-statistics control, not a semantic language
model or a claim of natural document coherence.
"""

from __future__ import annotations

import collections
import random

from corpus_backed_document_v5 import (
    BACKGROUND_RATE, BANK_SENTENCES, BG_STARTS, GENRES, PRIORS, SOURCE_HASH,
    TOKEN2ID, _background_distribution as bank_distribution, _task_prefix,
    site_label)
from genre_document_v4 import conditional_distribution as task_conditional
from joint_document_v1 import _normalize, _split


TRIGRAM = collections.defaultdict(collections.Counter)
BIGRAM = collections.defaultdict(collections.Counter)
UNIGRAM = collections.Counter()
for sentence in BANK_SENTENCES:
    history = ('<s>', '<s>')
    for word in sentence:
        TRIGRAM[history][word] += 1
        BIGRAM[history[-1]][word] += 1
        UNIGRAM[word] += 1
        history = (history[-1], word)
COMMON = dict(UNIGRAM.most_common(512))
assert len(COMMON) == 512 and '.' in COMMON


def _empirical(counter):
    n = sum(counter.values())
    return {word: count/n for word, count in counter.items()} if n else {}


COMMON_Q = _empirical(COMMON)


def _background_distribution(raw):
    if not raw:
        return bank_distribution(())
    if len(raw) >= 44:
        return {'.': 1.}
    a, b = (('<s>', '<s>')+tuple(raw))[-2:]
    tri = _empirical(TRIGRAM.get((a, b), {}))
    bi = _empirical(BIGRAM.get(b, {}))
    out = collections.Counter()
    for weight, q in ((.5, tri), (.25, bi), (.25, COMMON_Q)):
        for word, p in q.items():
            out[word] += weight*p
    if len(raw) < 6:
        out.pop('.', None)
    return _normalize(out)


def conditional_distribution(prefix, genre, regime='task_rich'):
    assert genre in GENRES and regime in BACKGROUND_RATE
    history, raw = _split(prefix)
    if raw and raw[0] in BG_STARTS:
        return _background_distribution(raw)
    task_prefix = _task_prefix(prefix)
    q_task = task_conditional(task_prefix, genre, regime)
    if not history or raw:
        return q_task
    p_bg = BACKGROUND_RATE[regime]
    q_bg = _background_distribution(())
    assert not (set(q_bg) & set(q_task))
    return _normalize({**{w: (1-p_bg)*p for w, p in q_task.items()},
                       **{w: p_bg*p for w, p in q_bg.items()}})


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
                mixed[word] = mixed.get(word, 0.)+self.posterior[genre]*p
        return _normalize(mixed), qs

    def observe(self, word, qs):
        assert word in TOKEN2ID
        self.posterior = _normalize({genre: self.posterior[genre]*
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
        q = qs[genre]
        word = rng.choices(tuple(q), weights=tuple(q.values()))[0]
        assert marginal.get(word, 0.) > 0
        words.append(word)
        labels.append(label)
        distributions.append(marginal)
        trace.observe(word, qs)
    return words, labels, distributions, genre

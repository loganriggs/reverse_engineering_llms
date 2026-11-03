"""V7 grammar fixture conditioned on visible document profiles and claims."""

from __future__ import annotations

import random

import genre_document_v7 as v7
from genre_document_v3 import _base_prefix
from genre_document_v4 import to_v3
from joint_document_v1 import _normalize, _split, visible_state
from stateful_genre_grammar_v8 import EXTRA_VOCAB, grammar


VOCAB = tuple(v7.TOKEN2ID) + tuple(sorted(EXTRA_VOCAB-set(v7.TOKEN2ID)))
TOKEN2ID = {word: i for i, word in enumerate(VOCAB)}


def _visible_facts(prefix):
    task_prefix = v7._task_prefix(prefix)
    canonical = _base_prefix(to_v3(task_prefix))
    history, _ = _split(canonical)
    state = visible_state(history)
    profiles = tuple(sorted(state['profiles'].items()))
    claims = tuple(sorted((item, attribute, speaker)
                          for (item, attribute), speaker
                          in state['claims'].items()))
    return profiles, claims


def conditional_distribution(prefix, genre, regime='task_rich'):
    assert genre in v7.GENRES and regime in v7.BACKGROUND_RATE
    history, raw = _split(prefix)
    if raw and raw[0] in v7.STARTS:
        return {word: p for word, p in grammar(genre, *_visible_facts(prefix))
                .next_distribution(raw).items() if word != '<eos>'}
    q_task = v7.task_conditional(v7._task_prefix(prefix), genre, regime)
    if not history or raw:
        return q_task
    p_bg = v7.BACKGROUND_RATE[regime]
    q_bg = grammar(genre, *_visible_facts(prefix)).next_distribution(())
    assert not (set(q_bg) & set(q_task))
    return _normalize({**{w: (1-p_bg)*p for w, p in q_task.items()},
                       **{w: p_bg*p for w, p in q_bg.items()}})


def next_distribution(prefix, regime='task_rich'):
    trace = v7.PosteriorTrace(regime, conditional_distribution, TOKEN2ID)
    for word in prefix:
        _, qs = trace.predict()
        trace.observe(word, qs)
    return trace.predict()[0]


def sample_sequence(seed, length=256, regime='task_rich'):
    rng = random.Random(seed)
    genre = rng.choices(tuple(v7.PRIORS),
                        weights=tuple(v7.PRIORS.values()))[0]
    trace = v7.PosteriorTrace(regime, conditional_distribution, TOKEN2ID)
    words, labels, distributions = [], [], []
    for _ in range(length):
        marginal, qs = trace.predict()
        word = rng.choices(tuple(qs[genre]),
                           weights=tuple(qs[genre].values()))[0]
        assert marginal.get(word, 0.) > 0
        words.append(word)
        labels.append(v7.site_label(trace.words))
        distributions.append(marginal)
        trace.observe(word, qs)
    return words, labels, distributions, genre

"""Joint task document with a corpus-sentence background branch.

The background is sampled from a finite, local FineWeb sentence bank. Its
prefix-trie distribution is exact for *that empirical bank*, while planted
task sentences retain the V4 exact oracle and semantic state. This is a
bridge/control, not a claim that an empirical sentence bank is a theory of
natural language or a generalizing next-token model.
"""

from __future__ import annotations

import functools
import hashlib
import random
import re

import torch
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_ROWS, DEFAULT_TOKENIZER
from genre_document_v4 import (GENRES, PRIORS, TOKEN2ID as TASK_IDS,
                               conditional_distribution as task_conditional,
                               site_label as task_label)
from joint_document_v1 import NAMES, _normalize, _split


WORD_OR_MARK = re.compile(
    r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[.,!?;:()\[\]\"']|\n")
SENTENCE_BREAK = re.compile(r'(?<=[.!?])\s+')
TASK_STARTS = set(NAMES) | {
    'Document', 'Notes', 'Report', 'Guide', 'Section', 'Part',
    'According', 'As', 'Filed', 'Who', 'Which', 'At', 'In',
    'After', 'Following', 'Step', 'Problem'}
BACKGROUND_RATE = {'task_rich': .20, 'reference': .70}


def _insert(root, sentence):
    node = root
    node['_n'] = node.get('_n', 0)+1
    for token in sentence:
        node = node.setdefault(token, {})
        node['_n'] = node.get('_n', 0)+1


@functools.lru_cache(maxsize=1)
def _bank():
    rows = torch.load(DEFAULT_ROWS, map_location='cpu', weights_only=True)
    assert tuple(rows.shape) == (480, 513)
    source_hash = hashlib.sha256(rows.numpy().tobytes()).hexdigest()
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    root = {}
    words = set()
    sentences = 0
    sentence_rows = []
    for row in rows:
        text = tokenizer.decode(row.tolist())
        for span in SENTENCE_BREAK.split(text):
            tokens = WORD_OR_MARK.findall(span)
            if not (7 <= len(tokens) <= 45 and
                    tokens[-1] in ('.', '?', '!')):
                continue
            if tokens[0] in TASK_STARTS or '.' in tokens[:-1]:
                continue
            # One sentence terminator for V4's visible-state splitter.
            sentence = tuple(tokens[:-1]+['.'])
            _insert(root, sentence)
            words.update(sentence)
            sentences += 1
            sentence_rows.append(sentence)
    assert sentences >= 5000
    assert root['_n'] == sentences
    return root, words, sentences, source_hash, tuple(sentence_rows)


BG_ROOT, BG_WORDS, BANK_SIZE, SOURCE_HASH, BANK_SENTENCES = _bank()
VOCAB = tuple(TASK_IDS) + tuple(sorted(BG_WORDS-set(TASK_IDS)))
TOKEN2ID = {word: i for i, word in enumerate(VOCAB)}
BG_STARTS = frozenset(BG_ROOT)-{'_n'}
assert not (BG_STARTS & TASK_STARTS)


def _background_distribution(raw):
    node = BG_ROOT
    for word in raw:
        node = node[word]
    return {word: child['_n']/node['_n']
            for word, child in node.items() if word != '_n'}


def _task_prefix(prefix):
    history, raw = _split(prefix)
    out = []
    for sentence in history:
        if sentence[0] not in BG_STARTS:
            out.extend(sentence)
    if raw and raw[0] not in BG_STARTS:
        out.extend(raw)
    return out


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


def site_label(prefix):
    history, raw = _split(prefix)
    if raw and raw[0] in BG_STARTS:
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
        mixture = {}
        for genre, q in qs.items():
            for word, p in q.items():
                mixture[word] = mixture.get(word, 0.)+self.posterior[genre]*p
        return _normalize(mixture), qs

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
        word = rng.choices(tuple(qs[genre]),
                           weights=tuple(qs[genre].values()))[0]
        assert marginal.get(word, 0.) > 0
        words.append(word)
        labels.append(label)
        distributions.append(marginal)
        trace.observe(word, qs)
    return words, labels, distributions, genre

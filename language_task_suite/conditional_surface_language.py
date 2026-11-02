"""Readable joint-rule language with a visible rule-strength intervention.

All rule weights remain positive, so changing only the initial mode token
preserves support of every generated body prefix. The same lexical, induction,
lead and place-skip candidates compete at each verb slot in every mode.
"""

import random

from surface_mixture_language import (DEFAULT_WEIGHTS,
                                      SurfaceMixtureLanguage, VOCAB as BODY_VOCAB,
                                      _history_and_clause)


MODES = ('Balanced', 'Lexical', 'Recall', 'Setting')
MODE_WEIGHTS = {
    'Balanced': dict(DEFAULT_WEIGHTS),
    'Lexical': dict(lexical=.70, induction=.10, lead=.12, place_skip=.08),
    'Recall': dict(lexical=.20, induction=.55, lead=.15, place_skip=.10),
    'Setting': dict(lexical=.20, induction=.15, lead=.15, place_skip=.50),
}
PAIR_WEIGHTS = {
    'Balanced': dict(lexical=.49, induction=.49, lead=.01, place_skip=.01),
    'Lexical': dict(lexical=.79, induction=.19, lead=.01, place_skip=.01),
    'Recall': dict(lexical=.19, induction=.79, lead=.01, place_skip=.01),
    'Setting': dict(lexical=.49, induction=.49, lead=.01, place_skip=.01),
}
REGIMES = {'full': MODE_WEIGHTS, 'pair': PAIR_WEIGHTS}
VOCAB = BODY_VOCAB + MODES + (':',)
assert len(VOCAB) == len(set(VOCAB))
TOKEN2ID = {token: i for i, token in enumerate(VOCAB)}
ID2TOKEN = {i: token for token, i in TOKEN2ID.items()}


class ConditionalSurfaceLanguage:
    def __init__(self, seed=0, mode_probs=None, regime='full'):
        self.rng = random.Random(seed)
        if regime not in REGIMES:
            raise ValueError(regime)
        self.regime = regime
        self.mode_weights = REGIMES[regime]
        self.mode_probs = dict(mode_probs or {mode: 1.0 for mode in MODES})
        assert set(self.mode_probs) == set(MODES)
        assert min(self.mode_probs.values()) >= 0 and sum(self.mode_probs.values()) > 0
        self.delegates = {mode: SurfaceMixtureLanguage(
            seed=seed+100+i, weights=self.mode_weights[mode])
            for i, mode in enumerate(MODES)}
        self.vocab = VOCAB
        self.token2id = TOKEN2ID
        self.id2token = ID2TOKEN

    def _tokens(self, prefix):
        return [ID2TOKEN[i] if isinstance(i, int) else i for i in prefix]

    def next_distribution(self, prefix):
        tokens = self._tokens(prefix)
        if not tokens:
            z = sum(self.mode_probs.values())
            return {mode: self.mode_probs[mode]/z for mode in MODES
                    if self.mode_probs[mode] > 0}, 'mode'
        mode = tokens[0]
        if mode not in MODES:
            raise ValueError(f'Unknown mode {mode}')
        if len(tokens) == 1:
            return {':': 1.0}, 'header'
        assert tokens[1] == ':'
        body = tokens[2:]
        q = self.delegates[mode].next_distribution(body)
        _, clause = _history_and_clause(body)
        return q, ('mixed_verb' if len(clause) == 3 else 'frame')

    def candidates(self, prefix):
        tokens = self._tokens(prefix)
        return self.delegates[tokens[0]].candidates(tokens[2:])

    def sample_sequence(self, length=72):
        tokens, labels, qs = [], [], []
        for _ in range(length):
            q, label = self.next_distribution(tokens)
            token = self.rng.choices(list(q), weights=list(q.values()))[0]
            tokens.append(token)
            labels.append(label)
            qs.append(q)
        return [TOKEN2ID[token] for token in tokens], labels, qs

    def sample_batch(self, n, length=72):
        rows = [self.sample_sequence(length) for _ in range(n)]
        return ([row[0] for row in rows], [row[1] for row in rows],
                [row[2] for row in rows])


def render(ids):
    text = ' '.join(ID2TOKEN[i] if isinstance(i, int) else i for i in ids)
    for punctuation in (',', '.', ':'):
        text = text.replace(' '+punctuation, punctuation)
    return text

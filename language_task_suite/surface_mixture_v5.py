"""Joint narrative, object-place retrieval, and subject-count questions.

Counting is a distinct rule family from the v4 object/place binding task.
Count questions occur only when observed subjects have different mention
counts, so the visible name must affect the correct answer. Narrative verbs
and objects retain their competing exact-oracle rule mixtures.
"""

from collections import Counter

from surface_mixture_v2 import _frame, _indices, _split
from surface_mixture_v4 import (SurfaceMixtureV4, VOCAB_V4, _content,
                                _narrative_history)


EXTRA = ('How', 'many', 'times', 'main', 'actor') + tuple(str(i) for i in range(10, 33))
VOCAB_V5 = VOCAB_V4 + tuple(token for token in EXTRA if token not in VOCAB_V4)
TOKEN2ID_V5 = {token: i for i, token in enumerate(VOCAB_V5)}
ID2TOKEN_V5 = {i: token for token, i in TOKEN2ID_V5.items()}


def subject_counts(history):
    return Counter(clause[_indices(_frame(clause))[0]]
                   for clause in _narrative_history(history))


class SurfaceMixtureV5(SurfaceMixtureV4):
    def __init__(self, seed=0, count_rate=.12, **kwargs):
        super().__init__(seed=seed, **kwargs)
        assert 0 <= count_rate < 1
        self.count_rate = count_rate
        self.vocab = VOCAB_V5
        self.token2id = TOKEN2ID_V5
        self.id2token = ID2TOKEN_V5

    def _tokens(self, prefix):
        return [ID2TOKEN_V5[i] if isinstance(i, int) else i for i in prefix]

    def _start_distribution(self, history, after_newline):
        out = super()._start_distribution(history, after_newline)
        counts = subject_counts(history)
        if len(set(counts.values())) < 2:
            return out
        content_mass = 1.0 if after_newline else 1.0-self.newline_rate
        out = {token: (mass if token == '\n' else mass*(1-self.count_rate))
               for token, mass in out.items()}
        out['How'] = content_mass*self.count_rate
        return out

    def _count_distribution(self, history, content):
        pos = len(content)
        fixed = {1: 'many', 2: 'times', 3: 'was',
                 5: 'the', 6: 'main', 7: 'actor', 8: '?',
                 9: 'Answer', 10: ':', 12: '.'}
        if pos in fixed:
            return {fixed[pos]: 1.0}
        counts = subject_counts(history)
        if pos == 4:
            return {name: 1/len(counts) for name in counts}
        if pos == 11:
            answer = str(counts[content[4]])
            if answer not in self.token2id:
                raise ValueError(f'Count {answer} exceeds the answer vocabulary')
            return {answer: 1.0}
        raise ValueError(f'Bad count-question position {pos}')

    def next_distribution(self, prefix):
        history, raw = _split(self._tokens(prefix))
        content = _content(raw)
        if content and content[0] == 'How':
            return self._count_distribution(history, content)
        return super().next_distribution(prefix)

    def count_answer(self, prefix):
        history, raw = _split(self._tokens(prefix))
        content = _content(raw)
        if not content or content[0] != 'How' or len(content) != 11:
            raise ValueError('Not at a count answer slot')
        return str(subject_counts(history)[content[4]])

    def sample_sequence(self, length=72):
        tokens, labels, qs = [], [], []
        for _ in range(length):
            q = self.next_distribution(tokens)
            _, raw = _split(tokens)
            content = _content(raw)
            label = 'frame'
            if content:
                if content[0] == 'How' and len(content) == 11:
                    label = 'count_answer'
                elif content[0] == 'Where' and len(content) == 8:
                    label = 'question_answer'
                elif content[0] not in ('How', 'Where'):
                    frame = _frame(content)
                    if len(content) == _indices(frame)[1]:
                        label = 'mixed_verb'
                    else:
                        from surface_mixture_v3 import OBJECT_INDEX
                        if len(content) == OBJECT_INDEX[frame]:
                            label = 'mixed_object'
            token = self.rng.choices(list(q), weights=list(q.values()))[0]
            labels.append(label)
            qs.append(q)
            tokens.append(token)
        return [TOKEN2ID_V5[token] for token in tokens], labels, qs


def render(ids):
    text = ' '.join(ID2TOKEN_V5[i] if isinstance(i, int) else i for i in ids)
    for token in (',', '.', '?', ':'):
        text = text.replace(' '+token, token)
    return text.replace(' \n ', '\n')

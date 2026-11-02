"""Versioned structural variation around the exact v2 joint-verb rule.

Narrative frames keep their subject, verb, place, and event slots so the four
competing verb proposals remain exactly defined. Determiners vary, and some
sentences end with a short visible parenthetical. This is a controlled form
stress test, not a claim to match web text.
"""

from surface_mixture_v2 import (EVENTS, INTERIOR_PLACES, PEOPLE, PLACES, TOKEN2ID,
                                VOCAB, SurfaceMixtureV2, _frame, _indices,
                                _split, _tail, uniform)


EXTRA = ('this', 'that', '(', ')')
VOCAB_VARIED = VOCAB + tuple(token for token in EXTRA if token not in TOKEN2ID)
TOKEN2ID_VARIED = {token: i for i, token in enumerate(VOCAB_VARIED)}
ID2TOKEN_VARIED = {i: token for token, i in TOKEN2ID_VARIED.items()}
DETERMINER_SLOTS = {
    'lead': (4, 8, 11),
    'subject': (3, 7, 11),
    'at': (1, 6, 10),
    'when': (1, 7, 11),
    'in_place': (1, 5, 9),
    'during': (1, 5, 9),
}


def varied_tail(suffix, parenthetical_mass):
    if not suffix:
        return {'.': .35*(1-parenthetical_mass),
                ',': .30*(1-parenthetical_mass),
                'because': .35*(1-parenthetical_mass),
                '(': parenthetical_mass}
    if suffix[0] != '(':
        return _tail(suffix)
    if len(suffix) == 1:
        return {'near': .4, 'during': .3, 'according': .3}
    if len(suffix) <= 4:
        kind = suffix[1]
        if len(suffix) == 2:
            return {'to' if kind == 'according' else 'the': 1.}
        if len(suffix) == 3:
            return uniform({'near': PLACES, 'during': EVENTS,
                            'according': PEOPLE}[kind])
        return {')': .6, ',': .4}
    if suffix[4] == ')':
        assert len(suffix) == 5
        return {'.': 1.}
    assert suffix[4] == ','
    if len(suffix) == 5:
        return {'near': .4, 'during': .3, 'according': .3}
    kind = suffix[5]
    if len(suffix) == 6:
        return {'to' if kind == 'according' else 'the': 1.}
    if len(suffix) == 7:
        return uniform({'near': PLACES, 'during': EVENTS,
                        'according': PEOPLE}[kind])
    if len(suffix) == 8:
        return {')': 1.}
    assert len(suffix) == 9
    return {'.': 1.}


class SurfaceMixtureV2Varied(SurfaceMixtureV2):
    def __init__(self, seed=0, weights=None, subject_repeat_mass=.4,
                 subject_return_mass=0., determiner_variation=.5,
                 parenthetical_mass=.2):
        super().__init__(seed=seed, weights=weights,
                         subject_repeat_mass=subject_repeat_mass,
                         subject_return_mass=subject_return_mass)
        if not 0 <= determiner_variation <= 1:
            raise ValueError('determiner_variation must lie in [0, 1]')
        if not 0 <= parenthetical_mass <= 1:
            raise ValueError('parenthetical_mass must lie in [0, 1]')
        self.determiner_variation = determiner_variation
        self.parenthetical_mass = parenthetical_mass
        self.vocab = VOCAB_VARIED
        self.token2id = TOKEN2ID_VARIED
        self.id2token = ID2TOKEN_VARIED

    def _tokens(self, prefix):
        return [ID2TOKEN_VARIED[i] if isinstance(i, int) else i
                for i in prefix]

    def next_distribution(self, prefix):
        _, clause = _split(self._tokens(prefix))
        if clause:
            frame = _frame(clause)
            pos = len(clause)
            base_end = _indices(frame)[3]
            if frame == 'in_place' and pos == 2:
                return uniform(INTERIOR_PLACES)
            if pos in DETERMINER_SLOTS[frame]:
                p = self.determiner_variation
                return {'the': 1-p, 'this': p/2, 'that': p/2}
            if pos >= base_end:
                return varied_tail(clause[base_end:],
                                   self.parenthetical_mass)
        return super().next_distribution(prefix)

    def sample_sequence(self, length=72):
        tokens, labels, qs = [], [], []
        for _ in range(length):
            q = self.next_distribution(tokens)
            _, clause = _split(tokens)
            label = 'frame'
            if clause:
                frame = _frame(clause)
                pos = len(clause)
                if pos == _indices(frame)[1]:
                    label = 'mixed_verb'
                elif pos in DETERMINER_SLOTS[frame]:
                    label = 'determiner'
                elif pos >= _indices(frame)[3] and clause[_indices(frame)[3]:]:
                    suffix = clause[_indices(frame)[3]:]
                    if suffix[0] == '(' and len(suffix) in (4, 8):
                        label = 'parenthesis_close'
            token = self.rng.choices(list(q), weights=list(q.values()))[0]
            labels.append(label)
            qs.append(q)
            tokens.append(token)
        return ([TOKEN2ID_VARIED[token] for token in tokens], labels, qs)


def render(ids):
    words = [ID2TOKEN_VARIED[i] if isinstance(i, int) else i for i in ids]
    text = ' '.join(words)
    for token in (',', '.', ')'):
        text = text.replace(' '+token, token)
    return text.replace('( ', '(')

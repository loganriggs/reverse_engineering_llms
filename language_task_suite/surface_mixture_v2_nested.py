"""Joint-verb stream with a controlled, depth-two bracket stack task.

This is a versioned computational probe, not a natural-language realism claim.
The next closer depends on the most recently unclosed opener. Nested tails
remain inside the same sentence and leave the existing verb oracle intact.
"""

from surface_mixture_v2 import (EVENTS, PLACES, _frame, _indices, _split,
                                uniform)
from surface_mixture_v2_varied import (DETERMINER_SLOTS, VOCAB_VARIED,
                                       SurfaceMixtureV2Varied)


OPENERS = ('(', '[')
CLOSERS = {'(': ')', '[': ']'}
VOCAB_NESTED = VOCAB_VARIED + (('[', ']') if '[' not in VOCAB_VARIED else ())
TOKEN2ID_NESTED = {token: i for i, token in enumerate(VOCAB_NESTED)}
ID2TOKEN_NESTED = {i: token for token, i in TOKEN2ID_NESTED.items()}


def _filler(suffix, opener_index):
    offset = len(suffix)-opener_index-1
    if offset == 0:
        return {'near': .5, 'during': .5}
    if offset == 1:
        return {'the': 1.}
    if offset == 2:
        return uniform(PLACES if suffix[opener_index+1] == 'near' else EVENTS)
    raise ValueError('Filler is complete')


def nested_tail(suffix, inner_mass):
    """Exact next-token distribution for one or two visibly nested spans."""
    if not suffix or suffix[0] not in OPENERS:
        raise ValueError('Expected a nested-tail opener')
    if len(suffix) <= 3:
        return _filler(suffix, 0)
    outer_close = CLOSERS[suffix[0]]
    if len(suffix) == 4:
        return {outer_close: 1-inner_mass, ',': inner_mass}
    if suffix[4] == outer_close:
        if len(suffix) != 5:
            raise ValueError('Completed tail')
        return {'.': 1.}
    if suffix[4] != ',':
        raise ValueError('Expected comma or outer closer')
    if len(suffix) == 5:
        return {opener: .5 for opener in OPENERS}
    if suffix[5] not in OPENERS:
        raise ValueError('Expected inner opener')
    if len(suffix) <= 8:
        return _filler(suffix, 5)
    if len(suffix) == 9:
        return {CLOSERS[suffix[5]]: 1.}
    if suffix[9] != CLOSERS[suffix[5]]:
        raise ValueError('Mismatched inner closer')
    if len(suffix) == 10:
        return {outer_close: 1.}
    if suffix[10] != outer_close:
        raise ValueError('Mismatched outer closer')
    if len(suffix) == 11:
        return {'.': 1.}
    raise ValueError('Completed tail')


class SurfaceMixtureV2Nested(SurfaceMixtureV2Varied):
    def __init__(self, seed=0, weights=None, subject_repeat_mass=.4,
                 subject_return_mass=0., determiner_variation=.5,
                 nested_mass=.3, inner_mass=.6):
        super().__init__(seed=seed, weights=weights,
                         subject_repeat_mass=subject_repeat_mass,
                         subject_return_mass=subject_return_mass,
                         determiner_variation=determiner_variation,
                         parenthetical_mass=0.)
        if not 0 <= nested_mass <= 1 or not 0 <= inner_mass <= 1:
            raise ValueError('Nested and inner masses must lie in [0, 1]')
        self.nested_mass = nested_mass
        self.inner_mass = inner_mass
        self.vocab = VOCAB_NESTED
        self.token2id = TOKEN2ID_NESTED
        self.id2token = ID2TOKEN_NESTED

    def _tokens(self, prefix):
        return [ID2TOKEN_NESTED[i] if isinstance(i, int) else i
                for i in prefix]

    def next_distribution(self, prefix):
        _, clause = _split(self._tokens(prefix))
        if clause:
            frame = _frame(clause)
            base_end = _indices(frame)[3]
            if len(clause) >= base_end:
                suffix = clause[base_end:]
                if suffix and suffix[0] in OPENERS:
                    return nested_tail(suffix, self.inner_mass)
                if not suffix:
                    ordinary = super().next_distribution(prefix)
                    out = {token: mass*(1-self.nested_mass)
                           for token, mass in ordinary.items() if mass}
                    for opener in OPENERS:
                        out[opener] = out.get(opener, 0.)+self.nested_mass/2
                    return {token: mass for token, mass in out.items() if mass}
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
                base_end = _indices(frame)[3]
                if pos == _indices(frame)[1]:
                    label = 'mixed_verb'
                elif pos in DETERMINER_SLOTS[frame]:
                    label = 'determiner'
                elif pos >= base_end:
                    suffix = clause[base_end:]
                    if suffix and suffix[0] in OPENERS:
                        if len(suffix) == 9:
                            label = 'nested_close_inner'
                        elif len(suffix) == 10 or (len(suffix) == 4 and
                                                    self.inner_mass == 0):
                            label = 'nested_close_outer'
                        elif len(suffix) == 4:
                            label = 'nested_close_outer_optional'
            token = self.rng.choices(list(q), weights=list(q.values()))[0]
            labels.append(label)
            qs.append(q)
            tokens.append(token)
        return ([TOKEN2ID_NESTED[token] for token in tokens], labels, qs)


def render_nested(ids):
    words = [ID2TOKEN_NESTED[i] if isinstance(i, int) else i for i in ids]
    text = ' '.join(words)
    for token in (',', '.', ')', ']'):
        text = text.replace(' '+token, token)
    return text.replace('( ', '(').replace('[ ', '[')

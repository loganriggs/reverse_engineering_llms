"""Variable-span successor to the fixed-lag nested-bracket baseline.

Each span independently chooses one or two filler clauses. The outer closer
after an inner span therefore occurs at several lags, removing the simple
fixed-position copy solution of v2_nested. Depth remains at most two.
"""

from surface_mixture_v2 import _frame, _indices, _split
from surface_mixture_v2_nested import (CLOSERS, OPENERS,
                                       SurfaceMixtureV2Nested)
from surface_mixture_v2_varied import DETERMINER_SLOTS
from surface_mixture_v2 import EVENTS, PLACES, uniform


def _phrase_slot(suffix, start):
    offset = len(suffix)-start
    if offset == 0:
        return {'near': .5, 'during': .5}
    if offset == 1:
        return {'the': 1.}
    if offset == 2:
        return uniform(PLACES if suffix[start] == 'near' else EVENTS)
    raise ValueError('Phrase already complete')


def _span_state(suffix, start, extra_mass, inner_mass, allow_inner):
    opener = suffix[start]
    assert opener in OPENERS
    closer = CLOSERS[opener]
    if len(suffix) <= start+3:
        return 'q', _phrase_slot(suffix, start+1)
    end = start+4
    has_second = len(suffix) > end and suffix[end] == 'and'
    if has_second:
        if len(suffix) <= end+3:
            return 'q', _phrase_slot(suffix, end+1)
        end += 4
    if len(suffix) == end:
        if has_second:
            remainder = 1.
        else:
            remainder = 1-extra_mass
        out = {closer: remainder*(1-inner_mass if allow_inner else 1.)}
        if not has_second:
            out['and'] = extra_mass
        if allow_inner:
            out[','] = remainder*inner_mass
        return 'q', {key: mass for key, mass in out.items() if mass}
    terminal = suffix[end]
    if terminal == closer:
        return 'closed', end+1
    if not allow_inner or terminal != ',':
        raise ValueError('Expected a closer or nested-span comma')
    if len(suffix) == end+1:
        return 'q', {opener: .5 for opener in OPENERS}
    if suffix[end+1] not in OPENERS:
        raise ValueError('Expected inner opener')
    state, value = _span_state(suffix, end+1, extra_mass, 0., False)
    if state == 'q':
        return state, value
    inner_end = value
    if len(suffix) == inner_end:
        return 'q', {closer: 1.}
    if suffix[inner_end] != closer or len(suffix) != inner_end+1:
        raise ValueError('Expected outer closer')
    return 'closed', inner_end+1


def variable_nested_tail(suffix, extra_mass, inner_mass):
    state, value = _span_state(suffix, 0, extra_mass, inner_mass, True)
    if state == 'q':
        return value
    if len(suffix) == value:
        return {'.': 1.}
    raise ValueError('Completed sentence tail')


class SurfaceMixtureV2NestedVarlen(SurfaceMixtureV2Nested):
    def __init__(self, seed=0, weights=None, subject_repeat_mass=.4,
                 subject_return_mass=0., determiner_variation=.5,
                 nested_mass=.3, inner_mass=.6, extra_mass=.5):
        super().__init__(seed=seed, weights=weights,
                         subject_repeat_mass=subject_repeat_mass,
                         subject_return_mass=subject_return_mass,
                         determiner_variation=determiner_variation,
                         nested_mass=nested_mass, inner_mass=inner_mass)
        if not 0 <= extra_mass <= 1:
            raise ValueError('Extra filler mass must lie in [0, 1]')
        self.extra_mass = extra_mass

    def next_distribution(self, prefix):
        _, clause = _split(self._tokens(prefix))
        if clause:
            base_end = _indices(_frame(clause))[3]
            if len(clause) >= base_end:
                suffix = clause[base_end:]
                if suffix and suffix[0] in OPENERS:
                    return variable_nested_tail(suffix, self.extra_mass,
                                                self.inner_mass)
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
                        stack = []
                        for token in suffix:
                            if token in OPENERS:
                                stack.append(token)
                            elif token in CLOSERS.values():
                                assert stack and CLOSERS[stack.pop()] == token
                        if len(q) == 1 and next(iter(q)) in CLOSERS.values():
                            label = ('nested_close_inner' if len(stack) == 2
                                     else 'nested_close_outer')
                        elif any(token in q for token in CLOSERS.values()):
                            label = 'nested_close_optional'
            token = self.rng.choices(list(q), weights=list(q.values()))[0]
            labels.append(label)
            qs.append(q)
            tokens.append(token)
        return ([self.token2id[token] for token in tokens], labels, qs)

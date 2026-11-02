"""Joint visible-prefix verb, object, and variable-length nested-tail language.

This combines the existing exact oracles without changing their local rules.
The extra sampler is necessary because the v3 sampler uses the smaller v2
vocabulary and does not label nested closing positions.
"""

from surface_mixture_v2 import _frame, _indices, _split
from surface_mixture_v2_nested import CLOSERS, OPENERS
from surface_mixture_v2_nested_varlen import SurfaceMixtureV2NestedVarlen
from surface_mixture_v2_varied import DETERMINER_SLOTS
from surface_mixture_v3 import OBJECT_INDEX, SurfaceMixtureV3


class SurfaceMixtureV3NestedVarlen(SurfaceMixtureV3,
                                   SurfaceMixtureV2NestedVarlen):
    def __init__(self, seed=0, weights=None, object_weights=None,
                 object_vocab_size=None, subject_repeat_mass=.4,
                 subject_return_mass=0., determiner_variation=.5,
                 nested_mass=.3, inner_mass=.6, extra_mass=.5):
        super().__init__(
            seed=seed, weights=weights, object_weights=object_weights,
            object_vocab_size=object_vocab_size,
            subject_repeat_mass=subject_repeat_mass,
            subject_return_mass=subject_return_mass,
            determiner_variation=determiner_variation,
            nested_mass=nested_mass, inner_mass=inner_mass,
            extra_mass=extra_mass)

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
                elif pos == OBJECT_INDEX[frame]:
                    label = 'mixed_object'
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
        return [self.token2id[token] for token in tokens], labels, qs

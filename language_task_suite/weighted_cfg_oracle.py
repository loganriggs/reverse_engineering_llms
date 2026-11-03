"""Exact prefix filtering for small, proper, acyclic weighted CFGs.

This is a deliberately small reference implementation, not a replacement for
EarleyX on a state-split grammar with millions of productions. Rules expand
with their stated probabilities; ambiguity is retained by summing the mass of
all derivations consistent with the visible prefix. All grammar yields must
end in the explicit EOS terminal.
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from math import isclose


EOS = '<eos>'


class WeightedCFG:
    def __init__(self, rules, start='S'):
        self.rules = {
            lhs: tuple((float(prob), tuple(rhs)) for prob, rhs in choices)
            for lhs, choices in rules.items()
        }
        assert start in self.rules
        for lhs, choices in self.rules.items():
            assert choices and all(p > 0 for p, _ in choices), lhs
            assert isclose(sum(p for p, _ in choices), 1., abs_tol=1e-12), lhs
            for _, rhs in choices:
                if EOS in rhs:
                    assert lhs == start and rhs[-1] == EOS
        assert all(rhs and rhs[-1] == EOS
                   for _, rhs in self.rules[start])
        # Reject recursive grammars: the epsilon closure below must terminate.
        seen, active = set(), set()

        def visit(lhs):
            assert lhs not in active, 'recursive grammar requires Earley parsing'
            if lhs in seen:
                return
            active.add(lhs)
            for _, rhs in self.rules[lhs]:
                for sym in rhs:
                    if sym in self.rules:
                        visit(sym)
            active.remove(lhs)
            seen.add(lhs)

        visit(start)
        self.start = start
        self.terminals = frozenset(
            sym for choices in self.rules.values()
            for _, rhs in choices for sym in rhs if sym not in self.rules)
        assert EOS in self.terminals

    @lru_cache(maxsize=2048)
    def closure(self, stack):
        """Distribution of terminal-frontier stacks from one pending stack."""
        if not stack or stack[-1] not in self.rules:
            return {stack: 1.}
        lhs = stack[-1]
        out = defaultdict(float)
        for prob, rhs in self.rules[lhs]:
            next_stack = stack[:-1] + rhs[::-1]
            for frontier, continuation in self.closure(next_stack).items():
                out[frontier] += prob*continuation
        return dict(out)

    @lru_cache(maxsize=16384)
    def frontier(self, prefix):
        """Unnormalized posterior over pending stacks after a token prefix."""
        prefix = tuple(prefix)
        if not prefix:
            return self.closure((self.start,))
        previous = self.frontier(prefix[:-1])
        token = prefix[-1]
        out = defaultdict(float)
        for stack, mass in previous.items():
            if stack and stack[-1] == token:
                for next_stack, continuation in self.closure(stack[:-1]).items():
                    out[next_stack] += mass*continuation
        return dict(out)

    def prefix_probability(self, prefix):
        return sum(self.frontier(tuple(prefix)).values())

    def next_distribution(self, prefix):
        states = self.frontier(tuple(prefix))
        out = defaultdict(float)
        for stack, mass in states.items():
            if stack:
                out[stack[-1]] += mass
        total = sum(out.values())
        if total == 0:
            return {}
        return {word: mass/total for word, mass in out.items()}

    def sample(self, rng):
        stack, words = (self.start,), []
        while stack:
            sym, stack = stack[-1], stack[:-1]
            if sym in self.rules:
                choices = self.rules[sym]
                rhs = rng.choices([rhs for _, rhs in choices],
                                  weights=[p for p, _ in choices])[0]
                stack += rhs[::-1]
            else:
                words.append(sym)
        return tuple(words)

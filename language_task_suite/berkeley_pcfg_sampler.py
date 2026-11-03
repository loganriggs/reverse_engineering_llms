"""Read and sample a Berkeley text PCFG for *form diagnostics only*.

This is deliberately separate from the exact prefix oracle. A recursive
corpus-induced grammar needs a validated probabilistic prefix parser before
its samples can be used as an oracle-scored model training distribution.
"""

from __future__ import annotations

import ast
import random
from collections import defaultdict
from pathlib import Path


def read_berkeley_grammar(base_path):
    """Load Berkeley .grammar/.lexicon without evaluating source text."""
    base_path = Path(base_path)
    rules = defaultdict(list)
    skipped_self_loops = 0
    for line in base_path.with_suffix('.grammar').read_text().splitlines():
        parts = line.split()
        assert len(parts) in (4, 5) and parts[1] == '->', line
        lhs, rhs = parts[0], tuple(parts[2:-1])
        probability = float(parts[-1])
        if len(rhs) == 1 and rhs[0] == lhs:
            skipped_self_loops += 1
            continue
        if probability > 0:
            rules[lhs].append((probability, rhs))
    lexicon_rows = 0
    for line in base_path.with_suffix('.lexicon').read_text().splitlines():
        tag, word, raw_probs = line.split(maxsplit=2)
        probs = ast.literal_eval(raw_probs.replace('NaN', '0.0'))
        assert isinstance(probs, list)
        for i, prob in enumerate(probs):
            if prob > 0:
                rules[f'{tag}_{i}'].append((float(prob), (word,)))
        lexicon_rows += 1
    for lhs, choices in rules.items():
        total = sum(p for p, _ in choices)
        assert total > 0, lhs
        rules[lhs] = tuple((p/total, rhs) for p, rhs in choices)
    symbols = set(rules)
    undefined = {sym for choices in rules.values()
                 for _, rhs in choices for sym in rhs if sym not in symbols}
    # Lexical words are terminals by design; a terminal can have the same
    # spelling as a grammar symbol only if the grammar itself uses that name.
    assert 'ROOT_0' in rules
    return dict(rules), dict(lexicon_rows=lexicon_rows,
                             rule_count=sum(map(len, rules.values())),
                             nonterminal_count=len(rules),
                             skipped_self_loops=skipped_self_loops,
                             distinct_terminal_spellings=len(undefined))


def sample_sentence(rules, rng: random.Random, *, max_expansions=1000,
                    max_tokens=200):
    """Return (tokens, reason); failed bounds are logged by the caller."""
    stack = ['ROOT_0']
    words = []
    expansions = 0
    while stack:
        sym = stack.pop()
        if sym not in rules:
            words.append(sym)
            if len(words) > max_tokens:
                return None, 'max_tokens'
            continue
        expansions += 1
        if expansions > max_expansions:
            return None, 'max_expansions'
        choices = rules[sym]
        rhs = rng.choices([rhs for _, rhs in choices],
                          weights=[p for p, _ in choices])[0]
        stack.extend(rhs[::-1])
    return tuple(words), None

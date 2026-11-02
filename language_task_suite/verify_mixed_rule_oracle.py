"""Monte Carlo checks for the original mixed-rule next-token oracle."""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

from mixed_rule_oracle import (core_distribution, observable_distribution,
                               observable_distributions_for_sequence,
                               outside_span_distribution)

DEFAULT_SOURCE = '/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits'


def tv(observed, expected, n):
    return 0.5*sum(abs(observed.get(token, 0)/n-p)
                   for token, p in expected.items())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default=os.environ.get('ATTN_CIRCUITS_DIR', DEFAULT_SOURCE))
    p.add_argument('--draws', type=int, default=20000)
    p.add_argument('--output', default='language_task_suite/mixed_rule_oracle_checks.json')
    a = p.parse_args()
    sys.path.insert(0, str(Path(a.source).resolve()))
    from generator import LanguageGenerator, ALL_RULES, DEFAULT_MIXING
    gen = LanguageGenerator(ALL_RULES, mixing_weights=DEFAULT_MIXING,
                            mode='mixed', seed=800)
    contexts = [[], ['$'], ['a', 'B0', 'a'], ['g', 'd'],
                ['g', '#'], ['b', 'S0', 'b', '!'],
                ['e', 'B1', 'g', 'd', 'e']]
    rows = []
    for context in contexts:
        exact = core_distribution(gen, context)
        observed = Counter(gen._sample_next_token(context)[0]
                           for _ in range(a.draws))
        distance = tv(observed, exact, a.draws)
        rows.append(dict(context=context, support=len(exact),
                         total_variation=distance,
                         expected_top=sorted(exact.items(), key=lambda x: -x[1])[:5],
                         observed_top=observed.most_common(5)))
        assert distance < 0.03, (context, distance)
    # Simulate the outer branch explicitly on a controlled context.
    context = ['a', 'B0', 'a']
    exact = outside_span_distribution(gen, context)
    observed = Counter()
    for _ in range(a.draws):
        if gen.rng.random() < DEFAULT_MIXING['bracket']:
            token = gen._emit_bracket_sequence()[0][0]
        else:
            token = gen._sample_next_token(context)[0]
        observed[token] += 1
    bracket_tv = tv(observed, exact, a.draws)
    assert bracket_tv < 0.03, bracket_tv
    # Full-sequence check for a naturally reachable trigram/bigram collision.
    prefix = ['g', 'd']
    exact_visible = observable_distribution(gen, prefix, total_length=8)
    found = Counter()
    n_found = 0
    for _ in range(max(a.draws*5, 100_000)):
        ids, _ = gen.sample_sequence(8)
        tokens = [gen.id2token[i] for i in ids]
        if tokens[:2] == prefix:
            found[tokens[2]] += 1
            n_found += 1
    visible_tv = tv(found, exact_visible, n_found)
    assert n_found > 500 and visible_tv < 0.04, (n_found, visible_tv)

    # Force the initial token B1 (which is uniform noise at position 0), then
    # reject until the second token is '('. It can arise from either bracket
    # insertion or the ordinary noise path, so this checks hidden-state mixing.
    ambiguous = ['B1', '(']
    exact_ambiguous = observable_distribution(gen, ambiguous, total_length=8)
    ambiguous_counts = Counter()
    n_ambiguous = 0
    for _ in range(max(a.draws*25, 500_000)):
        if gen.rng.random() < DEFAULT_MIXING['bracket']:
            span, _ = gen._emit_bracket_sequence()
            second, third = span[0], span[1]
        else:
            second = gen._sample_next_token(['B1'])[0]
            if second == '(' and gen.rng.random() < DEFAULT_MIXING['bracket']:
                third = gen._emit_bracket_sequence()[0][0]
            elif second == '(':
                third = gen._sample_next_token(['B1', '('])[0]
            else:
                continue
        if second == '(':
            ambiguous_counts[third] += 1
            n_ambiguous += 1
    ambiguous_tv = tv(ambiguous_counts, exact_ambiguous, n_ambiguous)
    assert n_ambiguous > 1000 and ambiguous_tv < 0.04, (n_ambiguous, ambiguous_tv)
    ids, _ = gen.sample_sequence(64)
    sample_tokens = [gen.id2token[i] for i in ids]
    all_distributions = observable_distributions_for_sequence(gen, sample_tokens)
    for t in (1, 5, 15, 40, 63):
        slow = observable_distribution(gen, sample_tokens[:t], total_length=64)
        assert max(abs(slow.get(tok, 0)-all_distributions[t].get(tok, 0))
                   for tok in gen.vocab) < 1e-12
    result = dict(config=vars(a), core_rows=rows,
                  outside_span=dict(context=context, total_variation=bracket_tv,
                                    expected_open_probability=exact.get('(', 0),
                                    observed_open_probability=observed.get('(', 0)/a.draws),
                  observable_checks=dict(reachable_collision=dict(prefix=prefix,
                      n=n_found, total_variation=visible_tv,
                      expected=exact_visible, observed=dict(found)),
                      ambiguous_opener=dict(prefix=ambiguous,n=n_ambiguous,
                          total_variation=ambiguous_tv,
                          expected=exact_ambiguous,observed=dict(ambiguous_counts))),
                  caveat='Exact for the original LanguageGenerator.sample_sequence '
                         'at the specified complete sequence length, under its rules '
                         'and fixed sampling weights. This does not describe FastLanguageGenerator '
                         'or the priority-dispatch v3 generator.')
    Path(a.output).write_text(json.dumps(result, indent=2)+'\n')
    print('max core TV', max(row['total_variation'] for row in rows),
          'bracket-start TV', bracket_tv,
          'visible collision TV', visible_tv,
          'ambiguous opener TV', ambiguous_tv)


if __name__ == '__main__':
    main()

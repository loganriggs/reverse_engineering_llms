"""Exact-oracle and supported-edit audit for the joint three-task language."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from eval_surface_mixture_v2_edits import build_pairs as verb_pairs
from eval_surface_mixture_v2_nested_edits import build_pairs as bracket_pairs
from eval_surface_mixture_v3_edits import build_object_pairs
from surface_mixture_v2 import _frame, _indices, _split
from surface_mixture_v2_nested import CLOSERS, OPENERS, render_nested
from surface_mixture_v2_varied import SurfaceMixtureV2Varied
from surface_mixture_v3 import OBJECT_INDEX
from surface_mixture_v3_nested_varlen import SurfaceMixtureV3NestedVarlen


def independent_object_distribution(gen, prefix):
    candidates = gen.object_candidates(prefix)
    mass = sum(gen.object_weights[name] for name in candidates)
    result = defaultdict(float)
    for name, distribution in candidates.items():
        for token, probability in distribution.items():
            result[token] += gen.object_weights[name] * probability / mass
    return dict(result), len(candidates)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--length', type=int, default=128)
    parser.add_argument('--seed', type=int, default=20261102)
    parser.add_argument('--object-vocab-size', type=int, default=8)
    parser.add_argument('--edit-streams', type=int, default=64)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    kwargs = dict(object_vocab_size=args.object_vocab_size)
    gen = SurfaceMixtureV3NestedVarlen(seed=args.seed, **kwargs)
    base = SurfaceMixtureV2Varied(seed=args.seed)
    labels, active_object_rules, bracket_lags = Counter(), Counter(), Counter()
    examples = []
    for stream in range(args.n):
        ids, site_labels, sampled_qs = gen.sample_sequence(args.length)
        tokens = [gen.id2token[i] for i in ids]
        if stream < 5:
            examples.append(render_nested(ids))
        for t, (token, label, q) in enumerate(zip(tokens, site_labels, sampled_qs)):
            prefix = tokens[:t]
            assert q == gen.next_distribution(prefix)
            assert q.get(token, 0) > 0 and abs(sum(q.values()) - 1) < 1e-12
            labels[label] += 1
            if label == 'mixed_verb':
                assert q == base.next_distribution(prefix)
            elif label == 'mixed_object':
                _, clause = _split(prefix)
                assert len(clause) == OBJECT_INDEX[_frame(clause)]
                independently, active = independent_object_distribution(gen, prefix)
                assert q.keys() == independently.keys()
                assert all(abs(q[k] - independently[k]) < 1e-12 for k in q)
                active_object_rules[active] += 1
            elif label in ('nested_close_inner', 'nested_close_outer'):
                _, clause = _split(prefix)
                suffix = clause[_indices(_frame(clause))[3]:]
                stack = []
                for i, value in enumerate(suffix):
                    if value in OPENERS:
                        stack.append((i, value))
                    elif value in CLOSERS.values():
                        assert stack and CLOSERS[stack.pop()[1]] == value
                expected_depth = 2 if label.endswith('inner') else 1
                assert len(stack) == expected_depth
                assert q == {CLOSERS[stack[-1][1]]: 1.0}
                bracket_lags[(label, len(suffix) - stack[-1][0])] += 1
    edit_counts = Counter()
    for builder, seed in ((verb_pairs, 1), (build_object_pairs, 2),
                          (bracket_pairs, 3)):
        edit_gen = SurfaceMixtureV3NestedVarlen(seed=args.seed+seed, **kwargs)
        edits = builder(edit_gen, args.edit_streams, args.length-1)
        assert edits
        edit_counts.update(pair['kind'] for pair in edits)
    assert labels['mixed_verb'] and labels['mixed_object']
    assert labels['nested_close_inner'] and labels['nested_close_outer']
    assert len({lag for (label, lag) in bracket_lags
                if label == 'nested_close_outer'}) >= 3
    result = dict(config={**vars(args), 'output': str(args.output)},
                  vocab_size=len(gen.vocab), labels=dict(labels),
                  active_object_rules=dict(active_object_rules),
                  bracket_lags={f'{label}@{lag}': count
                                for (label, lag), count in bracket_lags.items()},
                  edit_counts=dict(edit_counts), examples=examples,
                  caveat='Every sampled prefix replays the exact oracle. '
                         'Object mixtures were independently recomputed, '
                         'and verb distributions match the parent oracle. '
                         'This validates the synthetic DGP, not natural-language realism.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: result[key] for key in
                      ('vocab_size', 'labels', 'active_object_rules',
                       'bracket_lags', 'edit_counts')}, indent=2))


if __name__ == '__main__':
    main()

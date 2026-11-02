"""Independent grammar, oracle, and causal-edit audit for nested spans."""

import argparse
import json
from collections import Counter
from pathlib import Path

from surface_mixture_v2 import EVENTS, PLACES, _frame, _indices, _split
from surface_mixture_v2_nested import (CLOSERS, OPENERS,
                                       SurfaceMixtureV2Nested, render_nested)
from surface_mixture_v2_varied import SurfaceMixtureV2Varied


def supported(gen, prefix):
    history = []
    for token in prefix:
        if gen.next_distribution(history).get(token, 0) <= 0:
            return False
        history.append(token)
    return True


def check_completed(suffix):
    if suffix[0] not in OPENERS:
        return None
    assert suffix[-1] == '.'
    assert suffix[1] in ('near', 'during') and suffix[2] == 'the'
    assert suffix[3] in (PLACES if suffix[1] == 'near' else EVENTS)
    if suffix[4] in CLOSERS.values():
        assert len(suffix) == 6 and suffix[4] == CLOSERS[suffix[0]]
        return 'single'
    assert len(suffix) == 12 and suffix[4] == ','
    assert suffix[5] in OPENERS
    assert suffix[6] in ('near', 'during') and suffix[7] == 'the'
    assert suffix[8] in (PLACES if suffix[6] == 'near' else EVENTS)
    assert suffix[9] == CLOSERS[suffix[5]]
    assert suffix[10] == CLOSERS[suffix[0]]
    return 'nested'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=1000)
    p.add_argument('--length', type=int, default=128)
    p.add_argument('--seed', type=int, default=20261131)
    p.add_argument('--nested-mass', type=float, default=.3)
    p.add_argument('--inner-mass', type=float, default=.6)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    gen = SurfaceMixtureV2Nested(seed=a.seed, nested_mass=a.nested_mass,
                                 inner_mass=a.inner_mass)
    base = SurfaceMixtureV2Varied(seed=a.seed, parenthetical_mass=0.)
    labels, tails, pairs = Counter(), Counter(), Counter()
    verb_sites = 0
    edit_counts = Counter()
    examples = []
    support_checks = 0
    for stream in range(a.n):
        ids, labs, qs = gen.sample_sequence(a.length)
        tokens = [gen.id2token[i] for i in ids]
        if stream < 5:
            examples.append(render_nested(ids))
        for t, (token, label, q) in enumerate(zip(tokens, labs, qs)):
            assert abs(sum(q.values())-1) < 1e-12, (stream, t, q)
            assert token in q and q[token] > 0
            prefix = tokens[:t]
            assert q == gen.next_distribution(prefix)
            _, clause = _split(prefix)
            labels[label] += 1
            if not clause:
                continue
            frame = _frame(clause)
            base_end = _indices(frame)[3]
            if label == 'mixed_verb':
                assert t and len(clause) == _indices(frame)[1]
                assert q == base.next_distribution(prefix)
                verb_sites += 1
            if len(clause) < base_end:
                continue
            suffix = clause[base_end:]
            if not suffix or suffix[0] not in OPENERS:
                continue
            start = len(prefix)-len(clause)+base_end
            if label == 'nested_close_inner':
                assert len(suffix) == 9 and q == {CLOSERS[suffix[5]]: 1.}
                edited = prefix.copy()
                edited[start+5] = next(x for x in OPENERS if x != suffix[5])
                relevant_edit = edited
                assert gen.next_distribution(edited) == {
                    CLOSERS[edited[start+5]]: 1.}
                edit_counts['inner_relevant'] += 1
                edited = prefix.copy()
                edited[start] = next(x for x in OPENERS if x != suffix[0])
                control_edit = edited
                assert gen.next_distribution(edited) == q
                edit_counts['inner_outer_opener_control'] += 1
            elif label == 'nested_close_outer':
                assert len(suffix) == 10 and q == {CLOSERS[suffix[0]]: 1.}
                edited = prefix.copy()
                edited[start] = next(x for x in OPENERS if x != suffix[0])
                relevant_edit = edited
                assert gen.next_distribution(edited) == {
                    CLOSERS[edited[start]]: 1.}
                edit_counts['outer_relevant'] += 1
                edited = prefix.copy()
                filler = PLACES if suffix[6] == 'near' else EVENTS
                edited[start+8] = next(x for x in filler if x != suffix[8])
                control_edit = edited
                assert gen.next_distribution(edited) == q
                edit_counts['outer_filler_control'] += 1
            if (label in ('nested_close_inner', 'nested_close_outer')
                    and support_checks < 400):
                for candidate in (relevant_edit, control_edit):
                    assert supported(gen, candidate)
                    support_checks += 1
        history, _ = _split(tokens)
        for clause in history:
            kind = check_completed(clause[_indices(_frame(clause))[3]:])
            if kind:
                tails[kind] += 1
                suffix = clause[_indices(_frame(clause))[3]:]
                if kind == 'nested':
                    pairs[suffix[0]+suffix[5]] += 1
    assert all(edit_counts[k] for k in ('inner_relevant', 'inner_outer_opener_control',
                                         'outer_relevant', 'outer_filler_control'))
    result = dict(config={**vars(a), 'output': str(a.output)},
                  vocab_size=len(gen.vocab), verb_sites=verb_sites,
                  labels=dict(labels), completed_tails=dict(tails),
                  nested_opener_pairs=dict(pairs),
                  causal_edit_counts=dict(edit_counts),
                  edited_prefix_support_checks=support_checks,
                  examples=examples,
                  caveat='Visible-prefix oracle and sampled-token support '
                         'checked at every position. Bracket closure and '
                         'edited-prefix targets are independently checked; '
                         'this controlled syntax is not FineWeb-like prose.')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: result[key] for key in
                     ('verb_sites','labels','completed_tails',
                      'nested_opener_pairs','causal_edit_counts')}, indent=2))


if __name__ == '__main__':
    main()

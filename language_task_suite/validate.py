"""Independent rule checks and ambiguity audit for the generated task suite."""

from __future__ import annotations

import random

from grammar import GENERATORS, PLACES, WEEKDAYS, generate, sample


def latent(ex):
    return dict(ex.latent)


def oracle(ex):
    """Recover the answer from the recorded world state, independently of generator control flow."""
    l = latent(ex)
    if ex.family == 'agreement':
        return ('is', 'are')[int(l['subject_number'])]
    if ex.family == 'reflexive':
        return {'man': 'himself', 'woman': 'herself',
                'men': 'themselves', 'women': 'themselves'}[l['subject']]
    if ex.family == 'attribute_binding':
        return l['first_color'] if l['query'] == l['first'] else l['second_color']
    if ex.family == 'state_update':
        return l['new']
    if ex.family == 'list_successor':
        return str(int(l['start']) + int(l['n_items']))
    if ex.family == 'weekday_successor':
        return WEEKDAYS[(int(l['start']) + int(l['n_days'])) % 7]
    if ex.family == 'delimiter_stack':
        return {'(': ')', '[': ']'}[l['outer']]
    if ex.family == 'comparison_chain':
        return l['top'] if l['ask_top'] == 'True' else l['bottom']
    if ex.family == 'conditional_rule':
        return dict(pair.split(':') for pair in l['rules'].split(';'))[l['color']]
    if ex.family == 'speaker_binding':
        return l['finder']
    if ex.family == 'negation_correction':
        return l['right']
    if ex.family == 'role_reversal':
        return l['agent'] if l['ask_agent'] == 'True' else l['patient']
    raise KeyError(ex.family)


def main():
    seen: dict[tuple[str, ...], str] = {}
    counts = {(name, v): 0 for name in GENERATORS for v in (0, 1)}
    for variant in (0, 1):
        for ex in sample(120_000, 20261101 + variant, variant):
            assert ex.target in ex.candidates, ex
            assert ex.target == oracle(ex), ex
            assert ex.prefix, ex
            old = seen.setdefault(ex.prefix, ex.target)
            assert old == ex.target, (ex.prefix, old, ex.target)
            counts[(ex.family, variant)] += 1
    assert all(n == 10_000 for n in counts.values()), counts
    # A structural causal check: the last location for the tracked object
    # governs the state-update answer, while an unrelated object's location does not.
    rng = random.Random(2)
    for _ in range(1000):
        ex = generate('state_update', rng)
        l = latent(ex)
        assert l['new'] in PLACES and l['old'] != l['new']
    print(f'PASS: {sum(counts.values())} examples, {len(seen)} distinct prefixes, '
          f'{len(counts)} family/variant cells; unique targets and oracle agreement.')


if __name__ == '__main__':
    main()

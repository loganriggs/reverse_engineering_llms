"""Check the readable mixed-rule DGP and compare crude surface statistics."""

import argparse
import json
from collections import Counter
from pathlib import Path

from compare_realism import summary
from surface_mixture_language import (ADJECTIVES, EVENTS, LEADS,
                                      OBJECTS, PEOPLE, PLACES, PLACE_PREPS,
                                      TIME_PREPS, VERBS, SurfaceMixtureLanguage,
                                      _history_and_clause, render)


def check_clauses(gen, ids):
    tokens = [gen.id2token[i] for i in ids]
    history, _ = _history_and_clause(tokens)
    for clause in history:
        assert (clause[0] in LEADS and clause[1] == ',' and
                clause[2] in PEOPLE and clause[3] in VERBS and
                clause[4] == 'the' and clause[5] in ADJECTIVES and
                clause[6] in OBJECTS and clause[7] in PLACE_PREPS and
                clause[8] == 'the' and clause[9] in PLACES and
                clause[10] in TIME_PREPS and clause[11] == 'the' and
                clause[12] in EVENTS and clause[-1] == '.')
        expected = {
            0: ['.'], 1: ['while', None, 'watches', '.'],
            2: [',', 'according', 'to', None, '.'],
            3: ['as', None, 'looks', 'on', '.'],
        }[LEADS.index(clause[0]) % 4]
        tail = clause[13:]
        assert len(tail) == len(expected)
        for observed, target in zip(tail, expected):
            assert observed in PEOPLE if target is None else observed == target


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=3000)
    p.add_argument('--length', type=int, default=70)
    p.add_argument('--draws', type=int, default=200000)
    p.add_argument('--output', default='language_task_suite/surface_mixture_audit.json')
    a = p.parse_args()
    gen = SurfaceMixtureLanguage(seed=4001)
    ids, labels, distributions = gen.sample_batch(a.n, a.length)
    active_counts = Counter()
    conflicts = 0
    induction_count = 0
    verb_count = 0
    for row, labs, qs in zip(ids, labels, distributions):
        check_clauses(gen, row)
        for t, (token_id, q, label) in enumerate(zip(row, qs, labs)):
            token = gen.id2token[token_id]
            assert abs(sum(q.values())-1) < 1e-12
            assert q.get(token, 0) > 0
            oracle = gen.next_distribution(row[:t])
            assert q == oracle
            if label != 'mixed_verb':
                continue
            candidates = gen.candidates(row[:t])
            active_counts[len(candidates)] += 1
            modes = {max(dist, key=dist.get) for dist in candidates.values()}
            conflicts += len(modes) > 1
            induction_count += 'induction' in candidates
            verb_count += 1
    # Full-sequence rejection sampling checks a naturally reachable prefix.
    prefix = ['Today', ',', 'Alice']
    expected = gen.next_distribution(prefix)
    mc_gen = SurfaceMixtureLanguage(seed=4002)
    observed = Counter()
    accepted = 0
    for _ in range(a.draws):
        sequence = mc_gen.sample_sequence(4)[0]
        words = [mc_gen.id2token[i] for i in sequence]
        if words[:3] == prefix:
            observed[words[3]] += 1
            accepted += 1
    assert accepted > 500
    tv = .5*sum(abs(observed.get(tok, 0)/accepted-p)
                 for tok, p in expected.items())
    assert tv < .05
    texts = [render(row) for row in ids]
    fineweb = json.loads(Path(__file__).with_name(
        'world_v2_surface_comparison.json').read_text())['fineweb_513_token_windows']
    result = dict(config=vars(a), vocabulary_size=len(gen.vocab),
                  n_verb_positions=verb_count,
                  active_rule_counts=dict(active_counts),
                  conflict_fraction=conflicts/verb_count,
                  induction_active_fraction=induction_count/verb_count,
                  controlled_prefix=render(prefix),
                  controlled_distribution=expected,
                  controlled_accepted=accepted,
                  controlled_frequency_tv=tv,
                  synthetic_surface=summary(texts), fineweb_surface=fineweb,
                  examples=texts[:5],
                  caveat='Exact synthetic oracle and grammar checks do not prove '
                         'human naturalness. FineWeb units are 513-token windows, '
                         'whereas synthetic units are 70-token streams; sentence '
                         'statistics are more comparable than per-text statistics.')
    Path(a.output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: result[key] for key in (
        'vocabulary_size', 'n_verb_positions', 'active_rule_counts',
        'conflict_fraction', 'induction_active_fraction',
        'controlled_frequency_tv', 'synthetic_surface')}, indent=2))


if __name__ == '__main__':
    main()

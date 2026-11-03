"""Replay exact short-form rules and compare their word frequencies."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from audit_surface_repetition import WORD
from surface_mixture_v2 import ADJECTIVES, PLACES, _split
from surface_mixture_v3 import SurfaceMixtureV3
from surface_mixture_v3_variable import (
    SurfaceMixtureV3Variable, _canonical_short, _is_short, render,
)


def _statistics(streams, nwords):
    counts = Counter()
    eligible = 0
    for tokens in streams:
        words = [w.lower() for w in WORD.findall(render(tokens))]
        if len(words) < nwords:
            continue
        eligible += 1
        counts.update(words[:nwords])
    rank = np.asarray(sorted(counts.values(), reverse=True), dtype=float)
    total = int(rank.sum())
    lo, hi = 10, min(100, len(rank))
    slope = (float(np.polyfit(np.log(np.arange(lo, hi+1)),
                np.log(rank[lo-1:hi]/total), 1)[0]) if hi > lo else None)
    return dict(eligible_windows=eligible, total_words=total,
                observed_types=len(counts), top1_mass=float(rank[0]/total),
                top10_mass=float(rank[:10].sum()/total),
                the_mass=counts['the']/total, a_mass=counts['a']/total,
                an_mass=counts['an']/total, hapax_types=sum(c == 1 for c in
                counts.values()), rank10_to100_slope=slope,
                top25=counts.most_common(25), rank_counts=rank.astype(int).tolist())


def _check_joint(generator, prefix):
    """Article choice marginalizes, rather than changes, the V3 object rule."""
    article_q = generator.next_distribution(prefix)
    person, verb = prefix[-2:]
    history, _ = _split(generator._normalized(prefix))
    canonical_history = [word for sentence in history for word in sentence]
    for suffix in ([person], [person, verb, 'the', ADJECTIVES[0]]):
        q_by_place = [SurfaceMixtureV3.next_distribution(
            generator, canonical_history +
            ['At', 'the', place, ','] + suffix)
            for place in PLACES[:2]]
        assert q_by_place[0] == q_by_place[1]
    for adjective in ADJECTIVES:
        base = generator._short_query(prefix, [person, verb, 'a', adjective])
        recovered = Counter()
        for article, article_mass in article_q.items():
            adj_q = generator.next_distribution(prefix + [article])
            if adjective not in adj_q:
                continue
            obj_q = generator.next_distribution(
                prefix + [article, adjective])
            for obj, obj_mass in obj_q.items():
                recovered[obj] += article_mass*adj_q[adjective]*obj_mass
        for obj, expected in base.items():
            actual = recovered[obj]
            assert abs(actual-expected/len(ADJECTIVES)) < 1e-10


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=480)
    p.add_argument('--length', type=int, default=256)
    p.add_argument('--words', type=int, default=100)
    p.add_argument('--seed', type=int, default=99851)
    p.add_argument('--short-mass', type=float, default=.55)
    p.add_argument('--reference', type=Path, default=Path(
        'language_task_suite/results/v14_zipf_seed99851_n480/'
        'v14_vs_fineweb_word_rank.json'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    generator = SurfaceMixtureV3Variable(seed=args.seed,
                                         short_mass=args.short_mass)
    streams, label_counts = [], Counter()
    short_clauses, long_clauses, replayed, joint_checks = 0, 0, 0, 0
    examples = []
    for _ in range(args.n):
        ids, labels, qs = generator.sample_sequence(args.length)
        tokens = [generator.id2token[i] for i in ids]
        label_counts.update(labels)
        for t, (token, stored_q) in enumerate(zip(tokens, qs)):
            fresh_q = generator.next_distribution(tokens[:t])
            assert fresh_q == stored_q and token in fresh_q
            assert abs(sum(fresh_q.values())-1) < 1e-10
            replayed += 1
            _, current = _split(tokens[:t])
            if (_is_short(current) and len(current) == 2 and
                    joint_checks < 8):
                _check_joint(generator, tokens[:t])
                joint_checks += 1
        history, _ = _split(tokens)
        for clause in history:
            if _is_short(clause):
                canonical = _canonical_short(clause)
                assert canonical[-1] == '.'
                short_clauses += 1
                if len(examples) < 4:
                    examples.append(render(clause))
            else:
                long_clauses += 1
        streams.append(tokens)

    # At zero short mass, the new sampler is distributionally and
    # seed-by-seed identical to V3; only diagnostic labels differ.
    base, zero = SurfaceMixtureV3(seed=773), SurfaceMixtureV3Variable(
        seed=773, short_mass=0.)
    for _ in range(20):
        base_ids, _, base_q = base.sample_sequence(128)
        zero_ids, _, zero_q = zero.sample_sequence(128)
        assert base_ids == zero_ids and base_q == zero_q

    fixed = SurfaceMixtureV3(seed=args.seed)
    fixed_streams = []
    for _ in range(args.n):
        ids, _, _ = fixed.sample_sequence(args.length)
        fixed_streams.append([fixed.id2token[i] for i in ids])

    frequencies = dict(variable_core=_statistics(streams, args.words),
                       fixed_core=_statistics(fixed_streams, args.words))
    if args.reference.exists():
        reference = json.loads(args.reference.read_text())
        if (reference['n_eligible'] == args.n and
                reference['summary']['FineWeb']['total_words'] ==
                args.n*args.words):
            for name in ('V14', 'FineWeb'):
                frequencies[name] = dict(reference['summary'][name],
                                         rank_counts=reference['ranks'][name])
    fig, ax = plt.subplots(figsize=(7, 4.5), layout='constrained')
    palette = dict(variable_core='#bb562b', fixed_core='#9b8c21',
                   V14='#9051aa', FineWeb='#3569a4')
    for name, info in frequencies.items():
        rank = np.asarray(info['rank_counts'], dtype=float)
        ax.loglog(np.arange(1, len(rank)+1), rank/rank.sum(),
                  lw=1, marker='.', ms=2, color=palette[name], label=name)
    ax.set(xlabel='Word frequency rank', ylabel='Fraction of matched words',
           title='48,000-word frequency profiles')
    ax.legend()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure = args.output.with_suffix('.png')
    fig.savefig(figure, dpi=180)
    plt.close(fig)

    result = dict(config={**vars(args), 'output': str(args.output),
                          'reference': str(args.reference)},
                  exact_prefixes_replayed=replayed,
                  zero_short_mass_v3_identical_streams=20,
                  article_object_joint_checks=joint_checks,
                  completed_short_clauses=short_clauses,
                  completed_long_clauses=long_clauses,
                  labels=dict(label_counts), examples=examples,
                  frequencies=frequencies, figure=str(figure),
                  caveat='V3 core-rule form pilot, not the V14 source/fact/'
                         'question stream. Matching word ranks alone does '
                         'not establish realistic language or task rates.')
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: value for key, value in result.items()
                      if key != 'frequencies'}, indent=2), flush=True)
    for name, info in frequencies.items():
        print(name, {key: value for key, value in info.items()
                     if key not in ('top25', 'rank_counts')}, flush=True)


if __name__ == '__main__':
    main()

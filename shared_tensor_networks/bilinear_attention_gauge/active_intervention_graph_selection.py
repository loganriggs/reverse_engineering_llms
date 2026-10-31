"""Choose graph-disambiguating internal interventions by predicted information.

The two graphs are externally identical. Candidate QK/OV factors and the
two graph families are supplied; detector vectors are fitted from noisy
external observations. The active rule ranks Gaussian input pairs by the
two candidate zero-patch predictions' squared disagreement divided by
measurement variance. It never reads internal-patch labels before ranking.
"""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from intervention_graph_selection import fit_detectors, predict
from shared_computation_identifiability import forward, parameters


HERE = Path(__file__).resolve().parent
SIZES = (1, 2, 4, 8, 16, 32)
GRAPHS = ('shared', 'duplicate')


def one_trial(rng, teacher, noise_fraction, n_observation, pool_size, d):
    p = parameters(int(rng.integers(0, 2**31)), d)
    q = rng.normal(size=(n_observation, d))
    k = rng.normal(size=(n_observation, d))
    external = forward(q, k, p, teacher)
    sigma = noise_fraction * float(np.std(external))
    assert sigma > 0
    measured = external + rng.normal(scale=sigma, size=external.shape)
    detectors = fit_detectors(q, k, measured, p)

    qi = rng.normal(size=(pool_size, d))
    ki = rng.normal(size=(pool_size, d))
    candidate = {graph: predict(qi, ki, p, graph, detectors, 'zero')
                 for graph in GRAPHS}
    # KL between two Gaussian measurement laws with equal isotropic variance
    # is squared mean separation divided by 2 sigma^2.
    predicted_kl = np.sum((candidate['shared'] -
                           candidate['duplicate'])**2, axis=1) / (2*sigma*sigma)
    active = np.argsort(-predicted_kl, kind='stable')
    random = rng.permutation(pool_size)
    intact_shared = predict(qi, ki, p, 'shared', detectors, 'none')
    intact_duplicate = predict(qi, ki, p, 'duplicate', detectors, 'none')
    activity_score = np.sum(intact_shared**2 +
                            intact_duplicate**2, axis=1)
    activity = np.argsort(-activity_score, kind='stable')
    norm_score = np.linalg.norm(qi, axis=1)*np.linalg.norm(ki, axis=1)
    input_norm = np.argsort(-norm_score, kind='stable')
    target = forward(qi, ki, p, teacher, 'first_detector_zero')
    noisy_target = target + rng.normal(scale=sigma, size=target.shape)

    qt = rng.normal(size=(256, d))*np.geomspace(.4, 2.5, d)
    kt = rng.normal(size=(256, d))*np.geomspace(2.5, .4, d)
    heldout = forward(qt, kt, p, teacher, 'first_detector_half')
    half = {graph: predict(qt, kt, p, graph, detectors, 'half')
            for graph in GRAPHS}
    rows = []
    for n in SIZES:
        for strategy, order in (('active', active), ('random', random),
                                ('activity', activity),
                                ('input_norm', input_norm)):
            chosen_indices = order[:n]
            losses = {graph: float(np.mean(
                (candidate[graph][chosen_indices] -
                 noisy_target[chosen_indices])**2))
                for graph in GRAPHS}
            chosen = min(losses, key=losses.get)
            rows.append(dict(strategy=strategy, n_internal=n,
                correct=chosen == teacher,
                selected_heldout_half_patch_mse=float(
                    np.mean((half[chosen]-heldout)**2)),
                predicted_kl_sum=float(predicted_kl[chosen_indices].sum())))
    return rows


def run(seed=1201, trials=1000, n_observation=1024, pool_size=512, d=6):
    assert pool_size >= max(SIZES)
    rng = np.random.default_rng(seed)
    records = []
    for teacher in GRAPHS:
        for noise in (.5, 1., 2.):
            for trial in range(trials):
                result = one_trial(rng, teacher, noise, n_observation,
                                   pool_size, d)
                for item in result:
                    item.update(teacher=teacher, noise_fraction=noise,
                                trial=trial)
                records.extend(result)
            print(teacher, noise, trials, 'done', flush=True)
    summary = {}
    for teacher in GRAPHS:
        for noise in (.5, 1., 2.):
            for n in SIZES:
                selected = [r for r in records if r['teacher'] == teacher
                    and r['noise_fraction'] == noise and r['n_internal'] == n]
                arms = {s: [r for r in selected if r['strategy'] == s]
                        for s in ('active', 'random', 'activity', 'input_norm')}
                assert all(len(v)==trials for v in arms.values())
                summary[f'{teacher}__noise{noise}__n{n}'] = {
                    strategy: dict(
                        accuracy=float(np.mean([r['correct'] for r in rows])),
                        heldout_half_patch_mse=float(np.mean([
                            r['selected_heldout_half_patch_mse'] for r in rows])),
                        mean_predicted_kl_sum=float(np.mean([
                            r['predicted_kl_sum'] for r in rows])))
                    for strategy, rows in arms.items()}
    result = dict(description='Active versus random, high-output-activity, and high-input-norm selection from the same Gaussian input pool. Candidate graphs agree on all external inputs; active ranking uses only fitted-model candidate disagreement before any internal labels. Select graph from noisy zero-patch labels; test held-out half-strength patch under covariance shift.',
        seed=seed, trials=trials, n_observation=n_observation,
        pool_size=pool_size, d=d, sizes=SIZES, summary=summary)
    stem = f'active_intervention_graph_selection_s{seed}_t{trials}'
    path = HERE / f'{stem}.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    with gzip.open(HERE / f'{stem}_records.json.gz', 'wt') as stream:
        json.dump(records, stream, separators=(',', ':'))
    for teacher in GRAPHS:
        for noise in (.5, 1., 2.):
            print(teacher,noise,[(n,
                summary[f'{teacher}__noise{noise}__n{n}']['active']['accuracy'],
                summary[f'{teacher}__noise{noise}__n{n}']['activity']['accuracy'],
                summary[f'{teacher}__noise{noise}__n{n}']['input_norm']['accuracy'],
                summary[f'{teacher}__noise{noise}__n{n}']['random']['accuracy'])
                for n in SIZES],flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1201)
    parser.add_argument('--trials', type=int, default=1000)
    parser.add_argument('--pool-size', type=int, default=512)
    args = parser.parse_args()
    run(seed=args.seed, trials=args.trials, pool_size=args.pool_size)

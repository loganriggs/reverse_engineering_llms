"""Fresh-seed matched-prior confirmation; no Torch or live-run dependencies."""
import hashlib
import json
from pathlib import Path

import numpy as np

from noisy_joint_design import BUDGETS, D, P, PAIRS, features, order_design


def main():
    records = []
    errors = dict(augmented=0., scalar=0., covariance=0.)
    moment = .5 * sum((k - 1) / (D * (D - 1) * (k + 2)) for k in (3, 4))
    for seed in range(20):
        tr, pr, er, nr, sr = [np.random.default_rng(s) for s in
                              np.random.SeedSequence(282000 + seed).spawn(5)]
        theta = tr.normal(size=P)
        pool = np.zeros((8 * P, D))
        for k, (i, j) in enumerate(PAIRS):
            angles = pr.uniform(0, 2 * np.pi, 8)
            pool[8*k:8*k+8, i] = np.cos(angles)
            pool[8*k:8*k+8, j] = np.sin(angles)
        test = np.zeros((512, D))
        for k in range(512):
            support = er.choice(D, 3 + k % 2, replace=False)
            v = er.normal(size=len(support))
            test[k, support] = v / np.linalg.norm(v)
        x, xt = features(pool), features(test)
        np.testing.assert_allclose(np.linalg.norm(pool, axis=1), 1, atol=1e-14)
        np.testing.assert_allclose(np.linalg.norm(test, axis=1), 1, atol=1e-14)
        assert set(np.count_nonzero(pool, axis=1)) == {2}
        assert set(np.count_nonzero(test, axis=1)) == {3, 4}
        orders = {s: order_design(x, s, sr) for s in ('random', 'balanced', 'd_optimal')}
        assert all(len(set(o)) == max(BUDGETS) for o in orders.values())
        eps, single_eps = nr.normal(size=len(pool)), nr.normal(size=max(BUDGETS))
        for sigma in (.05, .2):
            y = x @ theta + sigma * eps
            for strategy in (*orders, 'singleton'):
                for budget in BUDGETS:
                    if strategy == 'singleton':
                        a, b = np.zeros((budget, P)), sigma * single_eps[:budget]
                    else:
                        ix = orders[strategy][:budget]
                        a, b = x[ix], y[ix]
                    ls, _, rank, _ = np.linalg.lstsq(a, b, rcond=1e-12)
                    gram = a.T @ a
                    inverse = np.linalg.inv(gram + sigma**2 * np.eye(P))
                    ridge = inverse @ a.T @ b
                    aug = np.linalg.lstsq(np.vstack([a, sigma * np.eye(P)]),
                                          np.r_[b, np.zeros(P)], rcond=1e-12)[0]
                    scalar = (a.T @ b) / ((a*a).sum(axis=0) + sigma**2)
                    covariance = sigma**2 * inverse
                    bias = inverse @ gram - np.eye(P)
                    replay_cov = bias @ bias.T + sigma**2 * inverse @ gram @ inverse
                    for name, first, second in [('augmented', ridge, aug),
                                                ('scalar', ridge, scalar),
                                                ('covariance', covariance, replay_cov)]:
                        errors[name] = max(errors[name], float(np.max(np.abs(first-second))))
                        np.testing.assert_allclose(first, second, atol=1e-11, rtol=1e-11)
                    for estimator, coef in [('least_squares', ls), ('matched_ridge', ridge)]:
                        records.append(dict(seed=seed, sigma=sigma, strategy=strategy,
                                            budget=budget, estimator=estimator, rank=int(rank),
                                            test_mse=float(np.mean((xt @ (coef-theta))**2)),
                                            bayes_prediction_risk=(float(moment*np.trace(covariance))
                                                                   if estimator == 'matched_ridge' else None)))
    assert len(records) == 1280
    summary = []
    for sigma in (.05, .2):
        for budget in BUDGETS:
            for estimator in ('least_squares', 'matched_ridge'):
                by = {s: [r for r in records if (r['sigma'], r['budget'], r['estimator'], r['strategy'])
                          == (sigma, budget, estimator, s)]
                      for s in ('random', 'balanced', 'd_optimal', 'singleton')}
                summary.append(dict(sigma=sigma, budget=budget, estimator=estimator,
                                    median_mse={s: float(np.median([r['test_mse'] for r in rows]))
                                                for s, rows in by.items()},
                                    design_wins={s: sum(a['test_mse'] < b['test_mse'] for a,b in
                                                        zip(by['d_optimal'],by[s]))
                                                 for s in ('random','balanced')}))
    sources = {s: hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in
               ('regularized_joint_design.py','REGULARIZED_JOINT_DESIGN_PLAN.md','noisy_joint_design.py')}
    result = dict(records=records, summary=summary, validation=errors,
                  monomial_second_moment=moment, sources=sources, numpy_version=np.__version__)
    root = Path('results/regularized_joint_design')
    root.mkdir(parents=True, exist_ok=False)
    (root/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(summary=[s for s in summary if s['estimator']=='matched_ridge'],
                         validation=errors),indent=2))


if __name__ == '__main__':
    main()

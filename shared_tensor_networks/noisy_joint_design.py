"""Known quadratic contrast family; CPU-only intervention-design control."""
import hashlib
import json
from pathlib import Path

import numpy as np


D = 8
PAIRS = [(i, j) for i in range(D) for j in range(i + 1, D)]
P = len(PAIRS)
BUDGETS = (14, 28, 56, 112)


def features(x):
    return np.stack([x[:, i] * x[:, j] for i, j in PAIRS], axis=1)


def order_design(x, strategy, rng):
    if strategy == 'random':
        return rng.permutation(len(x))[:max(BUDGETS)]
    if strategy == 'balanced':
        per_pair = [rng.permutation(np.arange(8) + 8 * i) for i in range(P)]
        return np.array([per_pair[i][round_] for round_ in range(4)
                         for i in rng.permutation(P)])
    inverse = np.eye(P) / 0.01
    used = np.zeros(len(x), dtype=bool)
    chosen = []
    for _ in range(max(BUDGETS)):
        scores = np.einsum('ij,jk,ik->i', x, inverse, x)
        scores[used] = -np.inf
        k = int(np.argmax(scores))
        v = inverse @ x[k]
        inverse -= np.outer(v, v) / (1 + x[k] @ v)
        chosen.append(k)
        used[k] = True
    return np.array(chosen)


def main():
    records = []
    forward_error = fit_error = 0.0
    for seed in range(20):
        teacher_rng, pool_rng, test_rng, noise_rng, select_rng = [
            np.random.default_rng(s) for s in np.random.SeedSequence(281000 + seed).spawn(5)]
        theta = teacher_rng.normal(size=P)
        pool = np.zeros((P * 8, D))
        for k, (i, j) in enumerate(PAIRS):
            angle = pool_rng.uniform(0, 2 * np.pi, 8)
            pool[k * 8:(k + 1) * 8, i] = np.cos(angle)
            pool[k * 8:(k + 1) * 8, j] = np.sin(angle)
        test = np.zeros((512, D))
        for k in range(len(test)):
            support = test_rng.choice(D, size=3 + k % 2, replace=False)
            v = test_rng.normal(size=len(support))
            test[k, support] = v / np.linalg.norm(v)
        np.testing.assert_allclose(np.linalg.norm(pool, axis=1), 1, atol=1e-14)
        np.testing.assert_allclose(np.linalg.norm(test, axis=1), 1, atol=1e-14)
        assert set(np.count_nonzero(pool, axis=1)) == {2}
        assert set(np.count_nonzero(test, axis=1)) == {3, 4}
        x, xt = features(pool), features(test)
        q = np.zeros((D, D))
        for value, (i, j) in zip(theta, PAIRS):
            q[i, j] = q[j, i] = value / 2
        independent = np.einsum('ij,jk,ik->i', test, q, test)
        forward_error = max(forward_error, float(np.max(np.abs(xt @ theta - independent))))
        np.testing.assert_allclose(xt @ theta, independent, atol=1e-13)
        assert not features(np.eye(D)).any()
        orders = {name: order_design(x, name, select_rng)
                  for name in ('random', 'balanced', 'd_optimal')}
        for order in orders.values():
            assert len(set(order)) == max(BUDGETS)
        # Common potential measurement noise for the shared candidate pool.
        eps = noise_rng.normal(size=len(pool))
        single_eps = noise_rng.normal(size=max(BUDGETS))
        for sigma in (0.05, 0.2):
            y = x @ theta + sigma * eps
            for strategy in (*orders, 'singleton'):
                for budget in BUDGETS:
                    if strategy == 'singleton':
                        a, b = np.zeros((budget, P)), sigma * single_eps[:budget]
                    else:
                        indices = orders[strategy][:budget]
                        a, b = x[indices], y[indices]
                    fitted, _, rank, _ = np.linalg.lstsq(a, b, rcond=1e-12)
                    denom = (a * a).sum(axis=0)
                    replay = np.divide(a.T @ b, denom, out=np.zeros(P), where=denom > 0)
                    fit_error = max(fit_error, float(np.max(np.abs(fitted - replay))))
                    np.testing.assert_allclose(fitted, replay, rtol=1e-10, atol=1e-10)
                    delta = fitted - theta
                    records.append(dict(seed=seed, sigma=sigma, strategy=strategy,
                                        budget=budget, rank=int(rank),
                                        test_mse=float(np.mean((xt @ delta)**2)),
                                        coefficient_mse=float(np.mean(delta**2))))
    summary = []
    for sigma in (0.05, 0.2):
        for budget in BUDGETS:
            by = {s: [r for r in records if r['sigma'] == sigma and
                      r['budget'] == budget and r['strategy'] == s]
                  for s in ('random', 'balanced', 'd_optimal', 'singleton')}
            summary.append(dict(sigma=sigma, budget=budget,
                                median_test_mse={s: float(np.median([r['test_mse'] for r in rows]))
                                                 for s, rows in by.items()},
                                median_rank={s: float(np.median([r['rank'] for r in rows]))
                                             for s, rows in by.items()},
                                d_optimal_wins={s: sum(a['test_mse'] < b['test_mse'] for a, b in
                                                      zip(by['d_optimal'], by[s]))
                                                for s in ('random', 'balanced')}))
    root = Path('results/noisy_joint_design')
    root.mkdir(parents=True, exist_ok=False)
    sources = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
               for name in ('noisy_joint_design.py', 'NOISY_JOINT_DESIGN_PLAN.md')}
    result = dict(sources=sources, numpy_version=np.__version__, records=records,
                  summary=summary, checks=dict(forward_error=forward_error, fit_error=fit_error))
    (root / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(summary=summary, checks=result['checks']), indent=2))


if __name__ == '__main__':
    main()

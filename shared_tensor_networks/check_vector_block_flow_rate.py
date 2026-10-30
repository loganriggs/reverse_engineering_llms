"""Check the vector-block quadratic theorem; no Torch/GPU or training dependencies."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm


def literal_block_map(h, indices, tau):
    """Solve selected affine block with all other coordinates frozen."""
    n = len(h)
    rest = np.array([i for i in range(n) if i not in indices])
    a = h[np.ix_(indices, indices)]
    b = h[np.ix_(indices, rest)]
    e = expm(-tau * a)
    out = np.eye(n)
    out[np.ix_(indices, indices)] = e
    out[np.ix_(indices, rest)] = np.linalg.solve(a, (e - np.eye(len(a))) @ b)
    return out


def radius(matrix):
    eig = np.linalg.eigvals(matrix)
    assert np.max(np.abs(eig.imag)) < 1e-10
    assert np.min(eig.real) > -1e-12
    return float(np.max(eig.real))


rng = np.random.default_rng(269101)
records = []
errors = dict(literal=0., similarity=0., spectrum=0., ode=0.)
for n in [3, 4, 6, 10]:
    for fixture in range(12):
        rotation, _ = np.linalg.qr(rng.normal(size=(n, n)))
        eigenvalues = np.geomspace(.05, 2., n)
        h = (rotation * eigenvalues) @ rotation.T
        split = 1 + fixture % (n - 1)
        first, second = np.arange(split), np.arange(split, n)
        p = np.diag([1.] * split + [0.] * (n - split))
        q = np.eye(n) - p
        root = (rotation * np.sqrt(eigenvalues)) @ rotation.T
        invroot = (rotation * (1 / np.sqrt(eigenvalues))) @ rotation.T
        u, v = root @ p @ root, root @ q @ root
        for tau in [.01, .1, .5, 2.]:
            a, b = literal_block_map(h, first, tau), literal_block_map(h, second, tau)
            c = b @ a
            direct = expm(-tau * q @ h) @ expm(-tau * p @ h)
            np.testing.assert_allclose(c, direct, rtol=2e-11, atol=2e-12)
            errors['literal'] = max(errors['literal'], float(np.max(abs(c - direct))))
            transformed = expm(-tau * v) @ expm(-tau * u)
            np.testing.assert_allclose(root @ c @ invroot, transformed, rtol=2e-11, atol=2e-12)
            errors['similarity'] = max(errors['similarity'], float(np.max(abs(root @ c @ invroot - transformed))))
            half = expm(-tau * v / 2)
            symmetric = half @ expm(-tau * u) @ half
            rho = float(np.linalg.eigvalsh(symmetric)[-1])
            for candidate in [c, a @ b, literal_block_map(h, second, tau / 2) @ a @ literal_block_map(h, second, tau / 2)]:
                discrepancy = abs(radius(candidate) - rho)
                assert discrepancy < 2e-11
                errors['spectrum'] = max(errors['spectrum'], discrepancy)
            joint = float(np.exp(-tau * np.linalg.eigvalsh(h)[0]))
            assert joint - 2e-12 <= rho < 1.
            if fixture == 0 and tau == .5:
                initial = rng.normal(size=n)
                left = solve_ivp(lambda t, x: -p @ h @ x, [0, tau], initial, method='DOP853', rtol=1e-12, atol=1e-13).y[:, -1]
                end = solve_ivp(lambda t, x: -q @ h @ x, [0, tau], left, method='DOP853', rtol=1e-12, atol=1e-13).y[:, -1]
                np.testing.assert_allclose(end, c @ initial, rtol=1e-10, atol=1e-11)
                errors['ode'] = max(errors['ode'], float(np.max(abs(end - c @ initial))))
            records.append(dict(n=n, split=split, fixture=fixture, tau=tau, alternating_radius=rho, joint_radius=joint))

# Nonzero cross-block coupling need not make the worst rate strictly worse.
h = np.array([[.1, 0., 0.], [0., 2., 1.], [0., 1., 2.]])
c = literal_block_map(h, np.array([2]), 1.) @ literal_block_map(h, np.array([0, 1]), 1.)
equality_example = dict(alternating_radius=radius(c), joint_radius=float(np.exp(-.1)), cross_block_coupling=1.)
np.testing.assert_allclose(equality_example['alternating_radius'], equality_example['joint_radius'], rtol=1e-13, atol=1e-13)
# Completely uncoupled blocks give identical full cycle maps.
h = np.diag([.1, .7, 2., 3.])
c = literal_block_map(h, np.array([2, 3]), .7) @ literal_block_map(h, np.array([0, 1]), .7)
np.testing.assert_allclose(c, expm(-.7 * h), rtol=1e-13, atol=1e-13)

out = Path('results/vector_block_flow_rate')
out.mkdir(exist_ok=False)
sources = ['check_vector_block_flow_rate.py', 'VECTOR_BLOCK_FLOW_RATE.md']
result = dict(scope='Two complementary vector blocks, fixed SPD quadratic, equal per-block active time', fixtures=len(records), max_errors=errors, coupled_equality_example=equality_example, records=records, sources={name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in sources})
(out / 'checks.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({key: value for key, value in result.items() if key not in ['records', 'sources']}), flush=True)

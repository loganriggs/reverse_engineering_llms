"""Independent NumPy forward + Gauss-Hermite quadrature audit of saved E1 weights."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


def forward(weights, prefix, x):
    p = {k.removeprefix(prefix): v for k, v in weights.items() if k.startswith(prefix)}
    l = (x @ p['A'].T) * (x @ p['B'].T)
    r = (x @ p['AR'].T) * (x @ p['BR'].T) if 'AR' in p else l
    return ((l @ p['C'].T) * (r @ p['D'].T)) @ p['w']


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('directory', type=Path)
    args = parser.parse_args(); root = args.directory
    protocol = json.loads((root/'protocol.json').read_text())['arguments']
    rows = list(csv.DictReader((root/'results.csv').open()))
    expected = len(protocol['teacher'].split(',')) * protocol['seeds'] * len(protocol['ntrain'].split(',')) * 3
    assert len(rows) == expected
    assert len({(r['teacher'], r['seed'], r['n'], r['student']) for r in rows}) == expected
    assert all(r['status'] == 'ok' and int(r['completed_steps']) == protocol['steps'] for r in rows)
    # Five nodes integrate degree <=9 in each coordinate: enough for squared quartics.
    node, weight = np.polynomial.hermite_e.hermegauss(5)
    weight /= np.sqrt(2*np.pi)
    indices = np.array(list(itertools.product(range(5), repeat=protocol['d'])))
    x = node[indices]; w = weight[indices].prod(axis=1)
    max_error, max_rms_error = 0., 0.
    hashes = {}
    for row in rows:
        path = root/row['evidence']; hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as weights:
            teacher = forward(weights, 'teacher_', x)
            student = forward(weights, 'student_', x)
            exact = float(w @ ((teacher-student)**2))
            rms = float(w @ teacher**2)
            params = sum(value.size for name, value in weights.items() if name.startswith('student_'))
            assert params == int(row['parameters'])
        error = abs(exact-float(row['exact_gaussian_mse'])) / max(abs(exact), 1e-14)
        assert error < 1e-8, (path, exact, row['exact_gaussian_mse'], error)
        assert abs(rms-1) < 1e-10
        max_error = max(max_error, error); max_rms_error = max(max_rms_error, abs(rms-1))
    result = dict(status='passed', rows=len(rows), independent_method='NumPy forward, 5-node tensor-product Gauss-Hermite quadrature',
                  max_relative_population_mse_error=max_error, max_teacher_second_moment_error=max_rms_error,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  results_sha256=hashlib.sha256((root/'results.csv').read_bytes()).hexdigest(), evidence_sha256=hashes)
    (root/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print({k:v for k,v in result.items() if k != 'evidence_sha256'})


if __name__ == '__main__':
    main()

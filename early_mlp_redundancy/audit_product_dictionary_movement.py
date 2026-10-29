"""Independent Torch/SVD audit of NumPy/QR movement and projection geometry."""
import json
import math
import torch
from fit_pca_encoder import V, sha
from measure_product_dictionary_movement import datasets


def main():
    torch.set_num_threads(1)
    rp = V / 'product_dictionary_movement.json'
    result = json.loads(rp.read_text())
    assert result['source_sha256'] == sha(V.parent / 'measure_product_dictionary_movement.py')
    assert result['protocol_sha256'] == sha(V / 'product_dictionary_movement_protocol.json')
    initial_entries = json.loads((V / 'product_dictionary_initial_states.json').read_text())['models']
    entries = json.loads((V / 'product_dictionary_training_results.json').read_text())['models']
    states = {}
    for entry in entries:
        init = next(e for e in initial_entries if e['seed'] == entry['seed'])
        assert sha(entry['path']) == entry['sha256'] and sha(init['path']) == init['sha256']
        s = {k: v.double() for k, v in torch.load(entry['path'], weights_only=True)['state'].items()}
        c0 = torch.load(init['path'], weights_only=True)['state']['coefficients'].double()
        U, _, _ = torch.linalg.svd(s['dictionary'], full_matrices=False)
        states[entry['seed'], entry['arm']] = (s, c0, U)
        expected = float((s['coefficients'] - c0).norm() / c0.norm())
        row = next(x for x in result['coefficients'] if (x['seed'], x['arm']) == (entry['seed'], entry['arm']))
        assert math.isclose(expected, row['relative_coefficient_change'], rel_tol=1e-10, abs_tol=1e-12)
    lookup = {(x['seed'], x['arm'], x['population'], x['document']): x for x in result['rows']}
    assert len(lookup) == len(result['rows']) == 132
    maximum = 0
    seen = set()
    for population, document, z, w in datasets():
        z, w = z.double(), w.double()
        first = next(iter(states.values()))[0]
        products = (z @ first['left'].T) * (z @ first['right'].T)
        for (seed, arm), (s, c0, U) in states.items():
            key = seed, arm, population, document
            assert key not in seen
            seen.add(key)
            row = lookup[key]
            q0 = torch.nn.functional.linear(torch.nn.functional.linear(products / s['rms'], c0), s['dictionary']) + s['offset']
            q = torch.nn.functional.linear(torch.nn.functional.linear(products / s['rms'], s['coefficients']), s['dictionary']) + s['offset']
            projected = (w - s['offset']) @ U @ U.T + s['offset']
            a, b, c = w - projected, q0 - projected, q - projected
            values = {'native_energy': w.square().sum(), 'irreducible_error': a.square().sum(),
                      'initial_error': (w - q0).square().sum(), 'final_error': (w - q).square().sum(),
                      'initial_projection_gap': b.square().sum(), 'final_projection_gap': c.square().sum(),
                      'correction_energy': (q - q0).square().sum(), 'initial_output_energy': q0.square().sum(),
                      'initial_cross': 2 * (a * b).sum(), 'final_cross': 2 * (a * c).sum()}
            energy = float(values['native_energy'])
            for name, value in values.items():
                error = abs(float(value) - row[name])
                assert error <= 1e-8 + 1e-10 * energy
                maximum = max(maximum, error / energy)
            assert row['positions'] == len(z)
            for prefix in ('initial', 'final'):
                gap = float(values[prefix + '_error'] - values['irreducible_error'] - values[prefix + '_projection_gap'])
                assert abs(gap) <= 1e-8 + 1e-10 * energy
                assert abs(float(values[prefix + '_cross'])) <= 1e-8 + 1e-10 * energy
        print('AUDIT_MOVEMENT', population, document, flush=True)
    assert seen == set(lookup)
    out = {'status': 'passed', 'source_sha256': sha(__file__), 'result_sha256': sha(rp), 'rows': len(seen),
           'max_energy_normalized_discrepancy': maximum,
           'scope': 'Independent Torch products, coefficient maps and SVD projection verify all132cases and4parameter changes. Shared data loader only; no new native forward.'}
    path = V / 'product_dictionary_movement_audit.json'
    assert not path.exists()
    path.write_text(json.dumps(out, indent=2) + '\n')
    print('PASSED', maximum, flush=True)


if __name__ == '__main__':
    main()

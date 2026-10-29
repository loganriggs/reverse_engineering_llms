"""M96 fixed-state movement and affine projection identities, no transformer calls."""
import json
import numpy as np
import torch
from fit_pca_encoder import A, V, sha


def datasets():
    ref = json.loads((V / 'native_input_reference.json').read_text())
    assert sha(ref['path']) == ref['sha256']
    r = torch.load(ref['path'], weights_only=True, map_location='cpu')
    receipt = json.loads((V / 'natural_token_context.json').read_text())
    assert sha(receipt['cache']) == receipt['cache_sha256']
    cached = torch.load(receipt['cache'], weights_only=True, map_location='cpu')
    yield 'train', -1, r['z'][r['indices']], cached['Y'][1][r['indices']]
    del cached
    natural = json.loads((V / 'product_dictionary_natural.json').read_text())
    assert len(natural['files']) == 32
    for file in natural['files']:
        assert sha(file['path']) == file['sha256']
        sample = torch.load(file['path'], weights_only=True, mmap=True)
        yield 'evaluation', sample['document'], sample['input'].reshape(-1, 1152), sample['write'].reshape(-1, 1152)


def main():
    torch.set_num_threads(1)
    pp = V / 'product_dictionary_movement_protocol.json'
    protocol = json.loads(pp.read_text())
    for name, digest in protocol['references'].items():
        assert sha(V / name) == digest
    initials = json.loads((V / 'product_dictionary_initial_states.json').read_text())['models']
    finals = json.loads((V / 'product_dictionary_training_results.json').read_text())['models']
    states, coefficients = {}, []
    for entry in finals:
        initial = next(x for x in initials if x['seed'] == entry['seed'])
        assert sha(entry['path']) == entry['sha256'] and sha(initial['path']) == initial['sha256']
        s = torch.load(entry['path'], weights_only=True)['state']
        t = torch.load(initial['path'], weights_only=True)['state']
        assert all(torch.equal(s[k], t[k]) for k in s if k != 'coefficients')
        s = {k: x.double().numpy() for k, x in s.items()}
        C0 = t['coefficients'].double().numpy()
        Q = np.linalg.qr(s['dictionary'], mode='reduced')[0]
        states[entry['seed'], entry['arm']] = (s, C0, Q)
        coefficients.append({'seed': entry['seed'], 'arm': entry['arm'],
                             'relative_coefficient_change': float(np.linalg.norm(s['coefficients'] - C0) / np.linalg.norm(C0))})
    rows = []
    for population, document, z, write in datasets():
        z, w = z.double().numpy(), write.double().numpy()
        first = next(iter(states.values()))[0]
        h = (z @ first['left'].T) * (z @ first['right'].T)
        energy = float(np.square(w).sum())
        assert energy > 0
        for (seed, arm), (s, C0, Q) in states.items():
            normalized = h / s['rms']
            q0 = (normalized @ C0.T) @ s['dictionary'].T + s['offset']
            q = (normalized @ s['coefficients'].T) @ s['dictionary'].T + s['offset']
            projection = ((w - s['offset']) @ Q) @ Q.T + s['offset']
            residual = w - projection
            initial_gap = q0 - projection
            final_gap = q - projection
            fields = {
                'native_energy': energy,
                'irreducible_error': float(np.square(residual).sum()),
                'initial_error': float(np.square(w - q0).sum()),
                'final_error': float(np.square(w - q).sum()),
                'initial_projection_gap': float(np.square(initial_gap).sum()),
                'final_projection_gap': float(np.square(final_gap).sum()),
                'correction_energy': float(np.square(q - q0).sum()),
                'initial_output_energy': float(np.square(q0).sum()),
                'initial_cross': float(2 * np.sum(residual * initial_gap)),
                'final_cross': float(2 * np.sum(residual * final_gap)),
            }
            for prefix in ('initial', 'final'):
                discrepancy = fields[prefix + '_error'] - fields['irreducible_error'] - fields[prefix + '_projection_gap']
                assert abs(discrepancy) <= 1e-8 + 1e-10 * energy
                assert abs(fields[prefix + '_cross']) <= 1e-8 + 1e-10 * energy
            rows.append({'seed': seed, 'arm': arm, 'population': population, 'document': document,
                         'positions': len(z), **fields})
        print('MOVEMENT', population, document, flush=True)
    assert len(rows) == 132
    result = {'status': 'measured_pending_independent_audit', 'source_sha256': sha(__file__),
              'protocol_sha256': sha(pp), 'coefficients': coefficients, 'rows': rows,
              'scope': 'Float64 evaluation of stored float32 states against cached native writes. Training4092unique positions and evaluation4096positions separate. No new transformer forwards or fitting.'}
    path = V / 'product_dictionary_movement.json'
    assert not path.exists()
    path.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()

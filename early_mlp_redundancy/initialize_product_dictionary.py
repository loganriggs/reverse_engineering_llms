"""M95 training-only normalization and exact projected-native initialization."""
import json
import numpy as np
import torch
from fit_pca_encoder import A, V, sha


def main():
    torch.set_num_threads(1)
    out = V / 'product_dictionary_initial_states.json'
    assert not out.exists()
    design_path = V / 'product_dictionary_featurizer_design.json'
    design = json.loads(design_path.read_text())
    for name, digest in design['source_hashes'].items():
        assert sha(V / name) == digest, name
    ref = json.loads((V / 'native_input_reference.json').read_text())
    assert sha(ref['path']) == ref['sha256']
    r = torch.load(ref['path'], weights_only=True, map_location='cpu')
    indices = r['indices'].numpy()
    assert len(indices) == len(np.unique(indices)) == 4092
    assert indices.min() >= 0 and indices.max() < 4096
    z = r['z'][r['indices']].double().numpy()
    left, right, W = (r[k].double().numpy() for k in ('L', 'R', 'D'))
    products = (z @ left.T) * (z @ right.T)
    rms = np.sqrt(np.mean(products ** 2, axis=0))
    zero = rms == 0
    rms[zero] = 1
    stats = A / 'emlp_m95_product_rms.pt'
    assert not stats.exists()
    torch.save({'rms': torch.from_numpy(rms), 'indices': r['indices'],
                'zero_rms': torch.from_numpy(zero)}, stats)
    entries = []
    for start in design['starts']:
        assert sha(start['path']) == start['sha256']
        s = torch.load(start['path'], weights_only=True, map_location='cpu')['state']
        B = s['features.dictionary'].double().numpy()
        norms = np.linalg.norm(B, axis=0)
        assert np.all(norms > 0)
        B = B / norms
        singular = np.linalg.svd(B, compute_uv=False)
        assert singular[-1] > 1e-10 * singular[0]
        inverse = np.linalg.pinv(B, rcond=1e-10)
        center = s['output_mean'].double().numpy() + float(s['output_scale']) * s['features.bias'].double().numpy()
        C = (inverse @ W) * rms[None, :]
        offset = center - B @ (inverse @ center)
        exact = {'left': r['L'].double(), 'right': r['R'].double(),
                 'rms': torch.from_numpy(rms), 'dictionary': torch.from_numpy(B),
                 'offset': torch.from_numpy(offset), 'coefficients': torch.from_numpy(C)}
        state = {k: value.float() for k, value in exact.items()}
        path = A / f"emlp_m95_initial_{start['seed']}.pt"
        assert not path.exists()
        torch.save({'state': state, 'float64_state': exact, 'seed': start['seed']}, path)
        entries.append({'seed': start['seed'], 'path': str(path), 'sha256': sha(path),
                        'source': start, 'dictionary_min_singular': float(singular[-1]),
                        'trainable_parameters': C.size})
    receipt = {'status': 'initialized_pending_independent_audit',
               'design_sha256': sha(design_path), 'source_sha256': sha(__file__),
               'runtime_sha256': sha(V.parent / 'product_dictionary_featurizer.py'),
               'reference_sha256': ref['sha256'], 'training_positions': len(indices),
               'evaluation_positions_used': 0, 'zero_rms_count': int(zero.sum()),
               'statistics_path': str(stats), 'statistics_sha256': sha(stats), 'models': entries}
    out.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()

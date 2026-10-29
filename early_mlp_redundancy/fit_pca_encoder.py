"""M73 fixed train-only PCA fits. Run audit_pca_encoder.py before native scoring."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
V = ROOT / 'validation'
A = Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition')
STATE = A / 'emlp_m73_pca_state.npz'
RECEIPT = V / 'pca_encoder_fit.json'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read(name):
    return json.loads((V / name).read_text())


def main():
    assert not STATE.exists() and not RECEIPT.exists(), 'Refusing to overwrite frozen fit'
    p = read('pca_encoder_baseline_protocol.json')
    bindings = {'least_squares_evaluation_comparison.json': 'reference_results_sha256',
                'least_squares_factorial.json': 'reference_factorial_sha256',
                'least_squares_natural.json': 'reference_natural_sha256',
                'natural_token_context_cohort.json': 'natural_cohort_sha256',
                'natural_token_context.json': 'natural_cache_receipt_sha256'}
    for name, key in bindings.items():
        assert sha(V / name) == p[key], name
    receipt = read('natural_token_context.json')
    assert sha(receipt['cache']) == receipt['cache_sha256']
    cohort = read('natural_token_context_cohort.json')
    train_hashes = {r['document_hash'] for r in cohort['records'][:32]}
    eval_hashes = {r['document_hash'] for r in cohort['records'][32:]}
    assert train_hashes.isdisjoint(eval_hashes)
    data = torch.load(receipt['cache'], weights_only=True, map_location='cpu')
    unique = data['unique_indices']
    indices = unique[data['documents'][unique] < 32].numpy().astype(np.int64)
    x = data['Y'][1][indices].double().numpy()
    mean = x.mean(axis=0)
    centered = x - mean
    covariance = centered.T @ centered / len(x)
    eigenvalues, basis = np.linalg.eigh(covariance)
    eigenvalues = eigenvalues[::-1].copy()
    basis = basis[:, ::-1].copy()
    assert np.isfinite(basis).all() and eigenvalues[127] > 0
    np.savez(STATE, indices=indices, mean=mean, basis=basis[:, :128],
             eigenvalues=eigenvalues, covariance=covariance)
    models = []
    for rank in (32, 128):
        models.append(dict(rank=rank, stored_affine_scalars=int(mean.size + mean.size * rank),
                           stored_affine_bytes_float64=int(8 * (mean.size + mean.size * rank)),
                           train_variance_explained=float(eigenvalues[:rank].sum()/eigenvalues.sum()),
                           cutoff_eigenvalue=float(eigenvalues[rank-1]),
                           next_eigenvalue=float(eigenvalues[rank]),
                           relative_cutoff_gap=float((eigenvalues[rank-1]-eigenvalues[rank])/eigenvalues[rank-1])))
    result = dict(status='fitted_pending_independent_audit', protocol_sha256=sha(V/'pca_encoder_baseline_protocol.json'),
                  source_sha256=sha(__file__), cache_sha256=receipt['cache_sha256'],
                  state_path=str(STATE), state_sha256=sha(STATE),
                  train_positions=len(indices), train_indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
                  training_document_hashes=sorted(train_hashes), evaluation_document_hashes=sorted(eval_hashes),
                  covariance_normalization='population: divide by number of training positions', models=models)
    RECEIPT.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()

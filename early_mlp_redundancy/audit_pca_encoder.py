"""Independent M73 fit audit: chunked torch moments and direct data SVD."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
V = ROOT / 'validation'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def main():
    receipt = json.loads((V/'pca_encoder_fit.json').read_text())
    assert sha(ROOT/'fit_pca_encoder.py') == receipt['source_sha256']
    assert sha(V/'pca_encoder_baseline_protocol.json') == receipt['protocol_sha256']
    assert sha(receipt['state_path']) == receipt['state_sha256']
    cache = json.loads((V/'natural_token_context.json').read_text())
    assert sha(cache['cache']) == receipt['cache_sha256'] == cache['cache_sha256']
    data = torch.load(cache['cache'], weights_only=True, map_location='cpu')
    state = np.load(receipt['state_path'])
    # Rebuild selection without using the fit's vectorized indexing expression.
    indices = np.array([int(i) for i in data['unique_indices'] if int(data['documents'][i]) < 32], dtype=np.int64)
    assert np.array_equal(indices, state['indices'])
    assert hashlib.sha256(indices.tobytes()).hexdigest() == receipt['train_indices_sha256']
    assert len(indices) == receipt['train_positions'] == len(set(indices.tolist()))
    cohort = json.loads((V/'natural_token_context_cohort.json').read_text())
    train = {r['document_hash'] for r in cohort['records'][:32]}
    test = {r['document_hash'] for r in cohort['records'][32:]}
    assert train.isdisjoint(test)
    assert sorted(train) == receipt['training_document_hashes']
    assert sorted(test) == receipt['evaluation_document_hashes']
    x = data['Y'][1][indices].double()
    mean = sum(chunk.sum(0) for chunk in x.split(113))/len(x)
    centered = x-mean
    cov = sum(chunk.T@chunk for chunk in centered.split(113))/len(x)
    mean_error = float((mean-torch.from_numpy(state['mean'])).abs().max())
    cov_error = float(torch.linalg.norm(cov-torch.from_numpy(state['covariance']))/torch.linalg.norm(cov))
    assert mean_error < 1e-9 and cov_error < 1e-12
    # Decompose the rectangular centered data directly, independently of eigh(C).
    _, singular, vt = torch.linalg.svd(centered, full_matrices=False)
    eig = singular.square()/len(x)
    assert np.allclose(eig.numpy(), state['eigenvalues'], rtol=1e-9, atol=1e-7)
    heldout = data['Y'][1][data['documents'] >= 32].double().numpy()
    rows = []
    errors = []
    for rank in (32, 128):
        u = state['basis'][:, :rank]
        projection = u@u.T
        ref = vt[:rank].numpy().T@vt[:rank].numpy()
        orth = float(np.linalg.norm(u.T@u-np.eye(rank)))
        idem = float(np.linalg.norm(projection@projection-projection))
        projector_error = float(np.linalg.norm(projection-ref))
        residual = float(np.linalg.norm(state['covariance']@u-u*state['eigenvalues'][:rank])/np.linalg.norm(state['covariance']@u))
        assert max(orth, idem, projector_error, residual) < 1e-9
        reconstructed = state['mean']+(heldout-state['mean'])@u@u.T
        independent = mean.numpy()+(heldout-mean.numpy())@ref
        assert np.allclose(reconstructed, independent, rtol=1e-9, atol=1e-8)
        cast = reconstructed.astype(np.float32)
        assert np.isfinite(cast).all()
        cast_rel2 = float(np.square(cast.astype(np.float64)-reconstructed).sum()/np.square(reconstructed).sum())
        assert cast_rel2 < 1e-12
        error = np.square(heldout-reconstructed).sum(axis=1)
        errors.append(error)
        rows.append(dict(rank=rank, orthogonality_error=orth, idempotence_error=idem,
                         independent_SVD_projector_error=projector_error, eigen_residual=residual,
                         cast_relative_squared_error=cast_rel2,
                         heldout_local_relative_squared_error=float(error.sum()/np.square(heldout).sum())))
    violations = int(np.sum(errors[1] > errors[0]+1e-8*np.maximum(errors[0],1)))
    assert violations == 0
    result = dict(status='passed', fit_receipt_sha256=sha(V/'pca_encoder_fit.json'), source_sha256=sha(__file__),
                  state_sha256=sha(receipt['state_path']), mean_max_absolute_error=mean_error,
                  covariance_relative_error=cov_error, heldout_positions=len(heldout),
                  nesting_violations=violations, models=rows,
                  scope='Fit and all natural heldout local projections only. Native downstream causal and retention scoring not yet run.')
    (V/'pca_encoder_preflight.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()

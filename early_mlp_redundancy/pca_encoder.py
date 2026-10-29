"""Frozen M73 affine PCA encoder; no input-dependent support selection."""
import numpy as np
import torch
from fit_pca_encoder import V, read, sha


class PCAEncoder:
    def __init__(self, rank):
        assert rank in (32, 128)
        fit = read('pca_encoder_fit.json')
        audit = read('pca_encoder_preflight.json')
        assert audit['status'] == 'passed'
        assert audit['fit_receipt_sha256'] == sha(V/'pca_encoder_fit.json')
        assert sha(fit['state_path']) == fit['state_sha256'] == audit['state_sha256']
        state = np.load(fit['state_path'])
        self.mean = state['mean']
        self.basis = state['basis'][:, :rank]
        self.rank = rank

    def reconstruct(self, write, variant=None):
        raw = write.detach().cpu().double().numpy()
        code = (raw-self.mean)@self.basis
        reconstruction = (self.mean+code@self.basis.T).astype(np.float32)
        support = torch.ones(code.shape, device=write.device)
        info = dict(rank_mean=float(self.rank), numerical_nonzero_mean=float((code != 0).sum(-1).mean()))
        return (torch.from_numpy(reconstruction).to(write.device), support,
                torch.from_numpy(code).to(write.device), info)

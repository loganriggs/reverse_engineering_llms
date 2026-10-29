"""M78 executable global affine function of normalized MLP input only."""
import json
import numpy as np
import torch
from fit_pca_encoder import sha,V

class AffineInputPredictor:
    def __init__(self):
        fit=json.loads((V/'affine_input_fit.json').read_text());audit=json.loads((V/'affine_input_fit_audit.json').read_text());assert audit['status']=='passed' and audit['fit_sha256']==sha(V/'affine_input_fit.json') and sha(fit['path'])==fit['sha256'];s=np.load(fit['path']);self.mx=s['mean_input'];self.my=s['mean_write'];self.A=s['coefficient'];self.rank=fit['rank']
    @torch.no_grad()
    def reconstruct(self,z,variant=None):
        code=z.detach().cpu().double().numpy()-self.mx
        pred=(self.my+code@self.A).astype(np.float32)
        return torch.from_numpy(pred).to(z.device),torch.ones(code.shape,device=z.device),torch.from_numpy(code).to(z.device),dict(rank_mean=float(self.rank),numerical_nonzero_mean=float((code!=0).sum(-1).mean()))
    def __call__(self,z):return self.reconstruct(z)[0]

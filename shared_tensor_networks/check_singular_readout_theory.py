"""Analytic Hessian, Pythagorean decomposition, and independent feature-noise toy."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def main():
    torch.set_default_dtype(torch.float64);rng=np.random.default_rng(90418);rows=[]
    for eps in [0.,1e-5,1e-4,1e-3,.01,.1,1.]:
        a=torch.tensor([.3,-.7],requires_grad=True)
        loss=lambda b:.5*((b[0]+b[1])**2+(eps*b[1]-1)**2)
        h=torch.autograd.functional.hessian(loss,a).detach().numpy()
        expected=np.array([[1.,1.],[1.,1.+eps*eps]])
        np.testing.assert_allclose(h,expected,rtol=1e-14,atol=1e-14)
        smallest=2*eps*eps/(2+eps*eps+np.sqrt(4+eps**4))
        numeric=float(np.linalg.eigvalsh(h)[0]);assert abs(numeric-smallest)<1e-14
        record=dict(epsilon=eps,smallest_hessian_eigenvalue=smallest,exact_fit_readout_norm=None,independent_noise_variance=None)
        if eps:
            beta=np.array([-1/eps,1/eps]);record['exact_fit_readout_norm']=float(np.linalg.norm(beta))
            record['independent_noise_variance']=float(1e-6*(beta@beta))
        rows.append(record)
    # Same fixed standardized noise draws at all eps: this tests the analytic
    # amplification factor, not many independent empirical confirmations.
    noise=rng.normal(size=(200000,2))*.001;checks=[]
    for eps in [.01,.1,1.]:
        beta=np.array([-1/eps,1/eps]);observed=float(np.mean((noise@beta)**2));predicted=1e-6*(beta@beta)
        relative=abs(observed-predicted)/predicted;assert relative<.02
        checks.append(dict(epsilon=eps,observed_noise_mse=observed,predicted_noise_mse=predicted,relative_error=relative))
    # Correlated common-mode noise cancels for this exact difference readout.
    common=np.repeat(noise[:,:1],2,axis=1)
    assert np.max(np.abs(common@np.array([-100.,100.])))<1e-15
    result=dict(status='passed',hessian_cases=rows,noise_checks=checks,common_mode_noise_cancels=True,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path=Path(__file__).parent/'results/singular_feature_paths/theory_checks.json'
    assert not path.exists();path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()

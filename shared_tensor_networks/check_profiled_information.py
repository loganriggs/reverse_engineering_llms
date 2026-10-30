"""Local information controls, not nonlinear training or recovery experiments."""
import hashlib
import json
from pathlib import Path
import numpy as np
from normalized_readout import normalized_design


def information(theta, x, W, tangent):
    phi, D, *_ = normalized_design(theta, x, 2)
    u, s, _ = np.linalg.svd(phi, full_matrices=False)
    assert s[-1] > s[0]*1e-12
    hidden = np.einsum('npk,pt,kl->ntl', D, W, tangent)
    profiled = hidden - np.einsum('nr,rm,mtk->ntk', u, u.T, hidden)
    A = profiled.reshape(-1, tangent.shape[1])
    # Independently eliminate the explicit readout Jacobian.
    R = np.zeros((len(x), W.shape[1], 3*W.shape[1]))
    for t in range(W.shape[1]):
        R[:, t, t*3:(t+1)*3] = phi
    H = hidden.reshape(-1, tangent.shape[1]); R = R.reshape(len(H), -1)
    direct = H - R @ np.linalg.lstsq(R, H, rcond=1e-12)[0]
    np.testing.assert_allclose(A, direct, rtol=1e-9, atol=1e-10)
    return A, A.T @ A


rows = np.eye(4)[[0, 2, 1, 3]]
theta = rows.ravel()
tangent = np.eye(16)[:, [4*i+j for i in range(4) for j in range(4) if rows[i,j] == 0]]
np.testing.assert_array_equal(tangent.T @ tangent, np.eye(12))
records=[]; monotonic=[]; noise_checks=[]
for seed in range(261101, 261106):
    rng=np.random.default_rng(seed); x=rng.normal(size=(256,4)); W=rng.normal(size=(3,6))
    matrices={}
    for n in [8,16,64,256]:
        for tasks in [1,3,6]:
            A,I=information(theta,x[:n],W[:,:tasks],tangent); matrices[n,tasks]=I
            singular=np.linalg.svd(A,compute_uv=False)
            rank=int(np.sum(singular>singular[0]*1e-10))
            records.append(dict(seed=seed,n=n,tasks=tasks,rank=rank,
                                smallest_singular=float(singular[-1]) if len(singular)==12 else 0.,largest_singular=float(singular[0]),
                                eigenvalues=np.linalg.eigvalsh(I).tolist()))
            if tasks==3 and n in [16,64,256]:
                assert rank==12
                # Exact linearized least-squares estimation under isotropic Gaussian noise.
                # No claim that finite-noise nonlinear training follows this estimator.
                draws=4000
                eps=rng.normal(size=(len(A),draws))
                estimates=np.linalg.pinv(A,rcond=1e-12)@eps
                predicted=float(np.sum(1/singular**2))
                observed=float(np.mean(np.sum(estimates**2,axis=0)))
                relative=abs(observed/predicted-1)
                assert relative<.12,(seed,n,relative)
                noise_checks.append(dict(seed=seed,n=n,draws=draws,noise_sd=1.,
                                         predicted_squared_error=predicted,observed_squared_error=observed,relative_error=relative))
    for tasks in [1,3,6]:
        for before,after in [(8,16),(16,64),(64,256)]:
            difference=matrices[after,tasks]-matrices[before,tasks]
            minimum=float(np.linalg.eigvalsh(difference)[0]); scale=max(1.,np.linalg.norm(matrices[after,tasks],2))
            assert minimum>=-1e-10*scale
            monotonic.append(dict(seed=seed,kind='inputs',before=before,after=after,tasks=tasks,minimum_eigenvalue=minimum))
    for n in [8,16,64,256]:
        for before,after in [(1,3),(3,6)]:
            difference=matrices[n,after]-matrices[n,before]
            minimum=float(np.linalg.eigvalsh(difference)[0]);scale=max(1.,np.linalg.norm(matrices[n,after],2))
            assert minimum>=-1e-10*scale
            monotonic.append(dict(seed=seed,kind='tasks',n=n,before=before,after=after,minimum_eigenvalue=minimum))

# Approach one retained, exactly rank-deficient eight-input fixture continuously.
certificate=json.loads(Path('results/sampled_information_certificate/certificate.json').read_text())
case=next(r for r in certificate['records'] if r['input_bound']==3 and r['seed']==260101 and r['n']==8)
assert case['profiled_rank']==11
base=np.array(case['input_points'],dtype=float)/3
full_case=next(r for r in certificate['records'] if r['input_bound']==30 and r['seed']==260101 and r['n']==8)
assert full_case['profiled_rank']==12
direction=np.array(full_case['input_points'],dtype=float)/30-base
near=[]
for delta in [0.,1e-1,1e-2,1e-3,1e-4,1e-5]:
    A,I=information(theta,base+delta*direction,np.eye(3),tangent)
    singular=np.linalg.svd(A,compute_uv=False)
    near.append(dict(delta=delta,smallest_singular=float(singular[-1]),largest_singular=float(singular[0]),
                     rank=int(np.sum(singular>singular[0]*1e-10))))
assert near[0]['rank']==11 and all(r['rank']==12 for r in near[1:])
assert near[-1]['smallest_singular']<near[1]['smallest_singular']/100

out=Path('results/profiled_information');out.mkdir(parents=True,exist_ok=True)
payload=dict(scope='Local linear information; no nonlinear optimizer or recovery claim',records=records,
             monotonicity=monotonic,linear_noise=noise_checks,near_degeneracy=near,
             source_sha256={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
                            ['check_profiled_information.py','normalized_readout.py','direct_bilinear_readout.py',
                             'results/sampled_information_certificate/certificate.json']})
(out/'checks.json').write_text(json.dumps(payload,indent=2)+'\n')
print('Information fixtures',len(records),'PSD differences',len(monotonic),'linear noise checks',len(noise_checks))
print('Maximum noise risk relative discrepancy',max(r['relative_error'] for r in noise_checks))
print('Near degeneracy',near)

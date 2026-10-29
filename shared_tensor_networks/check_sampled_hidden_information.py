import json
from pathlib import Path
import numpy as np
from normalized_readout import normalized_design

records=[];basis_checks=[]
for seed in range(253101,253104):
 rng=np.random.default_rng(seed);raw=rng.normal(size=(4,4));raw/=np.linalg.norm(raw,axis=1)[:,None];theta=raw.ravel()
 xall=rng.normal(size=(15,4));wall=rng.normal(size=(3,6))
 Z=(xall@raw[:2].T)*(xall@raw[2:].T);R=np.array([[1.2,.3],[-.2,.9]])
 def products(z):return np.stack([z[:,0]**2,z[:,0]*z[:,1],z[:,1]**2],axis=1)
 original=products(Z)@wall;changed=products(Z@R)
 refit=np.linalg.lstsq(changed,original,rcond=None)[0]
 basis_error=float(np.linalg.norm(changed@refit-original)/max(1,np.linalg.norm(original)));assert basis_error<1e-12
 basis_checks.append(dict(seed=seed,relative_prediction_error=basis_error))
 for n in [2,3,4,6,7,8,15]:
  x=xall[:n];phi,D,*_=normalized_design(theta,x,2);u,s,vh=np.linalg.svd(phi,full_matrices=False)
  rank=int(np.sum(s>s[0]*1e-11));assert rank==min(n,3)
  Q=u[:,:rank]
  for tasks in [1,3,6]:
   W=wall[:,:tasks];Y=phi@W;A=np.einsum('npk,pt->ntk',D,W)
   projected=A-np.einsum('nr,rm,mtk->ntk',Q,Q.T,A)
   jac=projected.reshape(n*tasks,16);sing=np.linalg.svd(jac,compute_uv=False)
   # Absolute scale references unprojected Jacobian to avoid counting roundoff when projection is zero.
   scale=max(1.,np.linalg.norm(A.reshape(n*tasks,16),2))
   ranks=[int(np.sum(sing>scale*cutoff)) for cutoff in [1e-9,1e-11]]
   bound=min(12,(n-rank)*np.linalg.matrix_rank(W));combined=min(bound,2*(n-2));assert all(r<=combined for r in ranks)
   direction=rng.normal(size=16);direction/=np.linalg.norm(direction)
   def residual(v):
    f=normalized_design(v,x,2)[0];coef=np.linalg.lstsq(f,Y,rcond=1e-12)[0]
    return (f@coef-Y).ravel()
   delta=1e-5;fd=(residual(theta+delta*direction)-residual(theta-delta*direction))/(2*delta)
   analytic=jac@direction;err=np.linalg.norm(fd-analytic)/max(1.,np.linalg.norm(analytic));assert err<1e-6,(seed,n,tasks,err)
   # Full-row interpolation also holds at a finite nearby hidden perturbation.
   interpolation=float(np.linalg.norm(residual(theta+.01*direction)))
   if n<=3:assert interpolation<1e-9
   records.append(dict(seed=seed,n=n,tasks=tasks,feature_rank=rank,readout_rank=int(np.linalg.matrix_rank(W)),rank_bound=int(bound),combined_bound=int(combined),observed_ranks=ranks,derivative_error=float(err),perturbed_residual_norm=interpolation))
out=Path('results/sampled_hidden_information');out.mkdir(exist_ok=True)
(out/'checks.json').write_text(json.dumps(dict(fixtures=len(records),records=records,basis_change_checks=basis_checks),indent=2)+'\n')
print('Fixtures',len(records),'max derivative error',max(r['derivative_error'] for r in records),'rank threshold disagreements',sum(len(set(r['observed_ranks']))>1 for r in records))
for n in [2,3,4,6,7,8,15]:print(n,[(t,sorted(set(r['observed_ranks'][0] for r in records if r['n']==n and r['tasks']==t))) for t in [1,3,6]])

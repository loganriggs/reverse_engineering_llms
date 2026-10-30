import hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from normalized_readout import normalized_design,objective

root=Path('results/local_noise_recovery');root.mkdir(exist_ok=False)
(root/'protocol.json').write_text(json.dumps({'source_sha256':{n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in ['LOCAL_NOISE_RECOVERY_PLAN.md','run_local_noise_recovery.py','normalized_readout.py','direct_bilinear_readout.py']}},indent=2)+'\n')
rows=np.eye(4)[[0,2,1,3]];theta=rows.ravel();Q=np.eye(16)[:,[4*i+j for i in range(4) for j in range(4) if rows[i,j]==0]]
records=[];derivative_checks=[]
for seed in range(262101,262106):
 rng=np.random.default_rng(seed);xall=rng.normal(size=(256,4));W=rng.normal(size=(3,3))
 for n in [16,64,256]:
  x=xall[:n];phi,D,*_=normalized_design(theta,x,2);y=phi@W
  u,s,_=np.linalg.svd(phi,full_matrices=False);assert s[-1]>1e-12*s[0]
  H=np.einsum('npk,pt,kl->ntl',D,W,Q);A=(H-np.einsum('nr,rm,mtk->ntk',u,u.T,H)).reshape(n*3,12)
  pinv=np.linalg.pinv(A,rcond=1e-12);assert np.linalg.matrix_rank(A)==12
  risk=float(np.sum(pinv**2));noise=rng.normal(size=(10,n,3));np.savez_compressed(root/f'fixture_{seed}_{n}.npz',x=x,W=W,noise=noise,A=A)
  for draw,e in enumerate(noise):
   for sigma in [.001,.0001,.00001]:
    target=y+sigma*e;cache={}
    def evaluate(v):
     if 'v' not in cache or not np.array_equal(v,cache['v']):
      residual,J,coef,_=objective(theta+Q@v,x,target,2,ridge=0.,gamma=0.,reduced=True)
      cache.update(v=v.copy(),r=residual,J=J@Q,W=coef)
     return cache
    if draw==0 and sigma==.001:
     point=np.linspace(-.001,.001,12);step=1e-6
     analytic=evaluate(point)['J'].copy()
     fd=np.column_stack([(evaluate(point+step*np.eye(12)[k])['r'].copy()-evaluate(point-step*np.eye(12)[k])['r'].copy())/(2*step) for k in range(12)])
     error=float(np.linalg.norm(fd-analytic)/max(1.,np.linalg.norm(analytic)));assert error<1e-6
     derivative_checks.append(dict(seed=seed,n=n,relative_error=error))
    start=time.perf_counter()
    fit=least_squares(lambda v:evaluate(v)['r'],np.zeros(12),jac=lambda v:evaluate(v)['J'],method='trf',ftol=1e-12,xtol=1e-12,gtol=1e-12,max_nfev=100)
    elapsed=time.perf_counter()-start;final=evaluate(fit.x);linear=pinv@(sigma*e.ravel())
    record=dict(seed=seed,n=n,draw=draw,sigma=sigma,status=int(fit.status),success=bool(fit.success),nfev=fit.nfev,njev=fit.njev,seconds=elapsed,
      coordinates=fit.x.tolist(),linear_coordinates=linear.tolist(),readout=final['W'].tolist(),squared_error=float(fit.x@fit.x),linear_squared_error=float(linear@linear),
      predicted_risk=sigma*sigma*risk,relative_linearization_error=float(np.linalg.norm(fit.x-linear)/max(np.linalg.norm(linear),1e-30)),
      loss=float(final['r']@final['r']/2),gradient_norm=float(np.linalg.norm(final['J'].T@final['r'])))
    assert np.isfinite(record['loss']) and np.isfinite(record['squared_error'])
    records.append(record)
  (root/'results.json').write_text(json.dumps(dict(records=records,derivative_checks=derivative_checks),indent=2)+'\n')
  print('Saved',seed,n,len(records),flush=True)
(root/'complete.json').write_text(json.dumps(dict(status='trained_not_audited',fits=len(records)))+'\n')

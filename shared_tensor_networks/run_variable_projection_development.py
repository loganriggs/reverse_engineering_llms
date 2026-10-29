import json,time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from direct_bilinear_readout import predict_jacobian
from variable_projection import reduced_residual
out=Path('results/variable_projection/development')
out.mkdir(exist_ok=False)
rows=[]
d,h,t=6,3,6;p=h*(h+1)//2;n=2*h*d
for seed in range(231101,231104):
 rng=np.random.default_rng(seed)
 teacher=np.r_[rng.normal(size=n)/np.sqrt(d),rng.normal(size=t*p)/np.sqrt(p)]
 x=rng.normal(size=(256,d));test=rng.normal(size=(4096,d))
 clean=predict_jacobian(teacher,x,h,False);scale=np.sqrt(np.mean(clean**2))
 teacher[n:]/=scale
 clean/=scale
 y=clean+.05*rng.normal(size=clean.shape)
 truth=predict_jacobian(teacher,test,h,False)
 theta=rng.normal(size=n)/np.sqrt(d)
 _,_,U=reduced_residual(theta,x,y,h)
 initial=np.r_[theta,U.T.ravel()]
 for method in ['joint','reduced']:
  cache={}
  def calc(v):
   if 'v' not in cache or not np.array_equal(v,cache['v']):
    if method=='joint':
     pred,J=predict_jacobian(v,x,h);r=pred-y
    else:r,J,_=reduced_residual(v,x,y,h)
    cache.update(v=v.copy(),r=r.ravel(),J=J.reshape(-1,len(v)))
   return cache
  start=time.perf_counter()
  fit=least_squares(lambda v:calc(v)['r'],initial if method=='joint' else theta,
                    jac=lambda v:calc(v)['J'],max_nfev=100,ftol=1e-10,xtol=1e-10,gtol=1e-10)
  elapsed=time.perf_counter()-start
  if method=='reduced':
   _,_,U=reduced_residual(fit.x,x,y,h);final=np.r_[fit.x,U.T.ravel()]
  else:final=fit.x
  pred=predict_jacobian(final,x,h,False)
  testpred=predict_jacobian(final,test,h,False)
  # Independent literal forward replay.
  A=final[:h*d].reshape(h,d);B=final[h*d:n].reshape(h,d);W=final[n:].reshape(t,p)
  z=(x@A.T)*(x@B.T)
  phi=np.stack([z[:,i]*z[:,j] for i in range(h) for j in range(i,h)],axis=1)
  np.testing.assert_allclose(phi@W.T,pred,atol=1e-12,rtol=1e-12)
  normal=np.linalg.norm(phi.T@(pred-y))
  relative_normal=normal/max(1.,np.linalg.norm(phi)*np.linalg.norm(pred-y))
  if method=='reduced':assert relative_normal<1e-10, (normal,relative_normal)
  row=dict(seed=seed,method=method,train_mse=float(np.mean((pred-y)**2)),
   fresh_sample_mse=float(np.mean((testpred-truth)**2)),nfev=fit.nfev,njev=fit.njev,
   seconds=elapsed,status=int(fit.status),readout_normal_residual=float(normal),readout_relative_normal_residual=float(relative_normal))
  np.savez(out/f'{seed}_{method}.npz',x=x,y=y,test=test,teacher=teacher,initial=initial,final=final)
  rows.append(row);print(row,flush=True)
(out/'results.json').write_text(json.dumps(rows,indent=2)+'\n')

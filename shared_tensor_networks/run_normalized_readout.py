import json,time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from normalized_readout import objective,normalized_design
from multitask import monomials,product_map
from experiment import gram
root=Path('results/normalized_readout/panel');root.mkdir(exist_ok=False)
d,h,t=8,4,10;k=64;p=10
keys=monomials(d);G=gram(keys)
def design(x):return np.prod(x[:,None,:]**np.array(keys)[None,:,:],axis=2)
rows=[]
for seed in range(232101,232106):
 rng=np.random.default_rng(seed);A=rng.normal(size=(h,d));B=rng.normal(size=(h,d))
 A/=np.linalg.norm(A,axis=1)[:,None];B/=np.linalg.norm(B,axis=1)[:,None]
 W=rng.normal(size=(14,p))/np.sqrt(p);K=product_map(A,B,keys);C=K@W.T
 scales=np.sqrt(np.sum(C*(G@C),axis=0));W/=scales[:,None];C/=scales
 x=rng.normal(size=(512,d));xa=rng.normal(size=(128,d))
 y=design(x)@C[:,:t]+.05*rng.normal(size=(512,t))
 ya=design(xa)@C[:,t:]+.05*rng.normal(size=(128,4))
 np.savez(root/f'{seed}_data.npz',x=x,y=y,xa=xa,ya=ya,teacher_A=A,teacher_B=B,teacher_W=W,truth=C)
 for init in [233001,233002]:
  irng=np.random.default_rng(init);raw=irng.normal(size=(8,8));raw/=np.linalg.norm(raw,axis=1)[:,None];theta=raw.ravel()
  _,_,U,_=objective(theta,x,y,h)
  for method in ['joint','reduced']:
   reduced=method=='reduced';v0=theta if reduced else np.r_[theta,U.T.ravel()];cache={}
   def calc(v):
    if 'v' not in cache or not np.array_equal(v,cache['v']):
     r,J,_,_=objective(v,x,y,h,reduced=reduced);cache.update(v=v.copy(),r=r,J=J)
    return cache
   start=time.perf_counter()
   fit=least_squares(lambda v:calc(v)['r'],v0,jac=lambda v:calc(v)['J'],max_nfev=200,ftol=1e-10,xtol=1e-10,gtol=1e-10)
   seconds=time.perf_counter()-start
   r,_,U,unit=objective(fit.x,x,y,h,reduced=reduced)
   F=product_map(unit[:h],unit[h:],keys);pred=F@U
   transfer=np.linalg.lstsq(design(xa)@F,ya,rcond=None)[0]
   err=pred-C[:,:t];te=F@transfer-C[:,t:]
   row=dict(seed=seed,init=init,method=method,nfev=fit.nfev,njev=fit.njev,status=int(fit.status),seconds=seconds,
     objective=float(.5*r@r),source_mse=float(np.sum(err*(G@err))/t),
     transfer_mse=float(np.sum(te*(G@te))/4),radial_max_error=float(np.max(np.abs(np.linalg.norm(fit.x[:k].reshape(8,8),axis=1)-1))))
   np.savez(root/f'{seed}_{init}_{method}.npz',initial=v0,final=fit.x,readout=U,features=F,transfer=transfer)
   rows.append(row);(root/'results.json').write_text(json.dumps(rows,indent=2)+'\n');print(row,flush=True)

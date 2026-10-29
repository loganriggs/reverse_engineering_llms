import hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from crossed_readout import objective
from multitask import monomials,product_map
from experiment import gram
src=Path('results/normalized_readout/panel');out=Path('results/crossed_readout');out.mkdir(exist_ok=False)
keys=monomials(8);G=gram(keys);rows=[]
for seed in range(232101,232106):
 data=np.load(src/f'{seed}_data.npz');x,y,xa,ya=[data[k] for k in ['x','y','xa','ya']];truth=data['truth']
 px=np.prod(xa[:,None,:]**np.array(keys)[None,:,:],axis=2)
 for init in [233001,233002]:
  theta=np.load(src/f'{seed}_{init}_reduced.npz')['initial']
  for name,normalized,ridge in [('unit_zero',True,0.),('raw_zero',False,0.),('raw_ridge',False,.001)]:
   _,_,U,_=objective(theta,x,y,4,ridge,normalized,True)
   for method in ['joint','reduced']:
    reduced=method=='reduced';v0=theta if reduced else np.r_[theta,U.T.ravel()];cache={}
    def calc(v):
     if 'v' not in cache or not np.array_equal(v,cache['v']):
      r,J,_,_=objective(v,x,y,4,ridge,normalized,reduced);cache.update(v=v.copy(),r=r,J=J)
     return cache
    start=time.perf_counter()
    fit=least_squares(lambda v:calc(v)['r'],v0,jac=lambda v:calc(v)['J'],max_nfev=200,ftol=1e-10,xtol=1e-10,gtol=1e-10)
    seconds=time.perf_counter()-start;r,_,U,ab=objective(fit.x,x,y,4,ridge,normalized,reduced)
    F=product_map(ab[:4],ab[4:],keys);adapt=np.linalg.lstsq(px@F,ya,rcond=None)[0]
    e=F@U-truth[:,:10];et=F@adapt-truth[:,10:]
    row=dict(seed=seed,init=init,condition=name,method=method,normalized=normalized,ridge=ridge,
     objective=float(.5*r@r),source_mse=float(np.sum(e*(G@e))/10),transfer_mse=float(np.sum(et*(G@et))/4),
     raw_max_norm=float(max(np.linalg.norm(fit.x[:64].reshape(8,8),axis=1))),nfev=fit.nfev,njev=fit.njev,status=int(fit.status),seconds=seconds)
    file=out/f'{seed}_{init}_{name}_{method}.npz';np.savez(file,initial=v0,final=fit.x,readout=U,features=F,transfer=adapt)
    row['checkpoint_sha256']=hashlib.sha256(file.read_bytes()).hexdigest();row['data_sha256']=hashlib.sha256((src/f'{seed}_data.npz').read_bytes()).hexdigest()
    rows.append(row);(out/'results.json').write_text(json.dumps(rows,indent=2)+'\n');print(row,flush=True)

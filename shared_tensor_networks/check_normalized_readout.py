import json,itertools
from pathlib import Path
import numpy as np
import torch
from normalized_readout import objective

torch.set_default_dtype(torch.float64);records=[]
for d,h in [(3,2),(6,3),(8,4)]:
 rng=np.random.default_rng(232000+d);x=rng.normal(size=(40,d));y=rng.normal(size=(40,3))
 k=2*h*d;p=h*(h+1)//2;theta=rng.normal(size=k)
 for reduced in [False,True]:
  v=theta if reduced else np.r_[theta,rng.normal(size=3*p)]
  def ref(v):
   raw=v[:k].reshape(2*h,d);norm=torch.linalg.vector_norm(raw,dim=1);unit=raw/norm[:,None]
   z=(torch.tensor(x)@unit[:h].T)*(torch.tensor(x)@unit[h:].T)
   phi=torch.stack([z[:,i]*z[:,j] for i,j in itertools.combinations_with_replacement(range(h),2)],dim=1)
   Z=torch.cat([phi,np.sqrt(.001)*torch.eye(p)]);Y=torch.cat([torch.tensor(y),torch.zeros(p,3)])
   U=torch.linalg.solve(Z.T@Z,Z.T@Y) if reduced else v[k:].reshape(3,p).T
   return torch.cat([(Z@U-Y).ravel(),norm-1])
  r,J,_,_=objective(v,x,y,h,reduced=reduced)
  vt=torch.tensor(v,requires_grad=True);rt=ref(vt).detach().numpy();JT=torch.autograd.functional.jacobian(ref,vt).numpy()
  error=np.linalg.norm(J-JT)/max(1.,np.linalg.norm(JT))
  assert error<1e-8,error
  np.testing.assert_allclose(r,rt,rtol=1e-9,atol=1e-9)
  direction=rng.normal(size=len(v));direction/=np.linalg.norm(direction)
  eps=1e-5;fd=(objective(v+eps*direction,x,y,h,reduced=reduced)[0]-objective(v-eps*direction,x,y,h,reduced=reduced)[0])/(2*eps)
  fd_error=np.linalg.norm(fd-J@direction)/max(1.,np.linalg.norm(J@direction));assert fd_error<1e-7
  records.append(dict(d=d,h=h,reduced=reduced,jacobian_error=error,finite_difference_error=fd_error))
path=Path('results/normalized_readout');path.mkdir(exist_ok=True)
(path/'checks.json').write_text(json.dumps(records,indent=2)+'\n');print(records)

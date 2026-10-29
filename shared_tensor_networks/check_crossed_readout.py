import itertools,json
from pathlib import Path
import numpy as np
import torch
from crossed_readout import objective
torch.set_default_dtype(torch.float64);rng=np.random.default_rng(234001);x=rng.normal(size=(50,6));y=rng.normal(size=(50,3));theta=rng.normal(size=36);records=[]
for normalized,ridge,reduced in itertools.product([False,True],[0.,.001],[False,True]):
 v=theta if reduced else np.r_[theta,rng.normal(size=18)]
 def ref(v):
  raw=v[:36].reshape(6,6);norm=torch.linalg.vector_norm(raw,dim=1);ab=raw/norm[:,None] if normalized else raw
  z=(torch.tensor(x)@ab[:3].T)*(torch.tensor(x)@ab[3:].T)
  phi=torch.stack([z[:,i]*z[:,j] for i,j in itertools.combinations_with_replacement(range(3),2)],dim=1)
  Z=torch.cat([phi,np.sqrt(ridge)*torch.eye(6)]);Y=torch.cat([torch.tensor(y),torch.zeros(6,3)])
  U=torch.linalg.solve(Z.T@Z,Z.T@Y) if reduced else v[36:].reshape(3,6).T
  r=(Z@U-Y).ravel()
  return torch.cat([r,norm-1]) if normalized else r
 r,J,_,_=objective(v,x,y,3,ridge,normalized,reduced);vt=torch.tensor(v,requires_grad=True)
 expected=torch.autograd.functional.jacobian(ref,vt).numpy();err=np.linalg.norm(J-expected)/max(1.,np.linalg.norm(expected))
 assert err<1e-8;np.testing.assert_allclose(r,ref(vt).detach().numpy(),rtol=1e-9,atol=1e-9)
 records.append(dict(normalized=normalized,ridge=ridge,reduced=reduced,jacobian_error=err))
Path('results/crossed_readout/checks.json').write_text(json.dumps(records,indent=2)+'\n');print('8 derivative checks passed; max',max(r['jacobian_error'] for r in records))

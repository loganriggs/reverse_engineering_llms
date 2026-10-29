import itertools,json
from pathlib import Path
import numpy as np
import torch
from scipy.linalg import expm
from readout_speed import calc,features,readout_flow,optimum

records=[]
for seed in range(248101,248104):
 rng=np.random.default_rng(seed);x=rng.normal(size=(32,8));y=rng.normal(size=(32,10));rows=rng.normal(size=(8,8));rows/=np.linalg.norm(rows,axis=1)[:,None];U=rng.normal(size=(10,10))
 loss,gr,gu=calc(rows,U,x,y)
 v=torch.tensor(np.r_[rows.ravel(),U.ravel()],dtype=torch.float64,requires_grad=True)
 raw=v[:64].reshape(8,8);unit=raw/torch.linalg.vector_norm(raw,dim=1)[:,None];xx=torch.tensor(x)
 z=(xx@unit[:4].T)*(xx@unit[4:].T);f=torch.stack([z[:,i]*z[:,j] for i,j in itertools.combinations_with_replacement(range(4),2)],1);u=v[64:].reshape(10,10)
 objective=((f@u-torch.tensor(y)).square().sum()+.001*u.square().sum())/64
 independent=torch.autograd.grad(objective,v)[0].numpy();error=np.max(np.abs(independent-np.r_[gr.ravel(),gu.ravel()]));assert error<1e-8
 f=features(rows,x)[0];H=(f.T@f+.001*np.eye(10))/32;rhs=f.T@y/32
 for tau in [.001,.1,1.]:
  # Augmented affine matrix exponential, avoiding a shared eigendecomposition/optimal readout implementation.
  block=np.zeros((20,20));block[:10,:10]=-H;block[:10,10:]=rhs
  expected=(expm(tau*block)@np.r_[U,np.eye(10)])[:10]
  actual=readout_flow(rows,U,x,y,tau);err=np.max(np.abs(actual-expected));assert err<1e-9
  assert calc(rows,actual,x,y,False)<=loss+1e-10
  records.append(dict(seed=seed,tau=tau,gradient_max_error=float(error),flow_max_error=float(err)))
out=Path('results/readout_speed_checks');out.mkdir(exist_ok=True);(out/'checks.json').write_text(json.dumps(records,indent=2)+'\n');print(records)

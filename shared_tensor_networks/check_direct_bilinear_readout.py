from pathlib import Path
import itertools
import numpy as np
import torch
from direct_bilinear_readout import predict_jacobian
from multitask import product_map,monomials
from run_svhn_standard import digest,write
rng=np.random.default_rng(224001);records=[]
for d,h in [(4,2),(6,3),(8,4)]:
 for heads in [1,h*(h+1)//2]:
  n=h*d;p=h*(h+1)//2;v=rng.normal(size=2*n+heads*p);x=rng.normal(size=(17,d));pred,J=predict_jacobian(v,x,h)
  def f(v):
   z=(torch.tensor(x)@v[:n].reshape(h,d).T)*(torch.tensor(x)@v[n:2*n].reshape(h,d).T)
   products=torch.stack([z[:,i]*z[:,j] for i in range(h) for j in range(i,h)],1)
   return products@v[2*n:].reshape(heads,p).T
  auto=torch.func.jacrev(f)(torch.tensor(v)).numpy();error=float(np.linalg.norm(auto-J)/max(1,np.linalg.norm(auto)));assert error<1e-12
  keys=monomials(d);phi=np.prod(x[:,None,:]**np.array(keys)[None,:,:],axis=2);C=product_map(v[:n].reshape(h,d),v[n:2*n].reshape(h,d),keys)@v[2*n:].reshape(heads,p).T
  np.testing.assert_allclose(pred,phi@C,atol=1e-9,rtol=1e-10);np.testing.assert_array_equal(pred,predict_jacobian(v,x,h,False))
  records.append(dict(d=d,h=h,heads=heads,jacobian_relative_error=error))
root=Path(__file__).parent
write(root/'results/direct_bilinear_readout_checks.json',dict(status='passed',records=records,source_sha256=digest(Path(__file__)),implementation_sha256=digest(root/'direct_bilinear_readout.py')))
print('Six direct-forward/coefficient/autograd checks passed.')

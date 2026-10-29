"""M102 executable mixed-interaction candidates."""
import json
import torch
from torch.nn import functional as F
from fit_pca_encoder import V,sha

class NativeRandomCandidate:
 def __init__(self,entry):
  self.entry=entry;self.kind=entry['kind'];self.variant='input';ref=json.loads((V/'native_input_reference.json').read_text());assert sha(ref['path'])==ref['sha256'];s=torch.load(ref['path'],weights_only=True);fit=json.loads((V/'native_random_subspaces.json').read_text());assert sha(fit['path'])==fit['sha256'];a=json.loads((V/'native_random_subspaces_audit.json').read_text());assert a['status']=='passed' and a['fit_sha256']==sha(V/'native_random_subspaces.json');self.basis=entry['basis'];self.rank=entry['rank'];self.L,self.R,self.D=[s[n].cuda().double() for n in ['L','R','D']];self.mean=s['mean'].cuda().double();self.constant=self.write(self.mean);self.J=(self.D*(self.R@self.mean))@self.L+(self.D*(self.L@self.mean))@self.R
  if self.basis not in ['native','shared_linear']:
   models=torch.load(fit['path'],weights_only=True)['models'];m=models[self.basis];self.E=m['encoder'][:,:self.rank].cuda();self.B=m['decoder'][:,:self.rank].cuda();assert torch.allclose(m['mean'],s['mean'],rtol=1e-12,atol=1e-12)
 def write(self,z):return F.linear(F.linear(z,self.L)*F.linear(z,self.R),self.D)
 @torch.no_grad()
 def reconstruct(self,z,write=None):
  if self.basis=='native':
   pred=F.linear(F.linear(z,self.L.to(z.dtype))*F.linear(z,self.R.to(z.dtype)),self.D.to(z.dtype));code=z.double()
  else:
   x=z.double()-self.mean;pred=self.constant+F.linear(x,self.J)
   if self.basis=='shared_linear':code=x[...,:0]
   else:code=x@self.E;pred=self.write(z.double())-self.write(x-code@self.B.T)
   pred=pred.to(z.dtype)
  info=dict(rank_mean=float(code.shape[-1]),numerical_nonzero_mean=float((code!=0).sum(-1).double().mean()),retained_native_products=0 if self.basis=='shared_linear' else 4608,output_dimension=1152)
  return pred,torch.ones_like(code),code,info
 def __call__(self,z):return self.reconstruct(z)[0]

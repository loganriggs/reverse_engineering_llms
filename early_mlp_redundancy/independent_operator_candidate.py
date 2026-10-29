"""Float64 NumPy oracle for projected-quadratic candidates; no native writes."""
import json
import numpy as np
import torch
from fit_pca_encoder import V,sha
from native_projected_quadratic import prediction,write

class IndependentOperatorCandidate:
 def __init__(self,entry):
  self.entry=entry;ref=json.loads((V/'native_input_reference.json').read_text());assert sha(ref['path'])==ref['sha256'];r=torch.load(ref['path'],weights_only=True);self.L,self.R,self.D=[r[n].double().numpy() for n in ['L','R','D']];self.mean=r['mean'].numpy();self.basis=entry['basis'];self.rank=entry['rank']
  if self.basis not in ['native','shared_linear']:
   fit=json.loads((V/'native_operator_metric_fits.json').read_text());assert sha(fit['path'])==fit['sha256'];m=torch.load(fit['path'],weights_only=True)['models'][self.basis];self.E=m['encoder'][:,:self.rank].numpy();self.B=m['decoder'][:,:self.rank].numpy()
  else:self.E=self.B=np.empty((1152,0))
 def prepare(self,z):return z.detach().cpu().double().numpy()
 def verify(self,x,prediction_tensor,support,code):
  if self.basis=='native':expected=write(self.L,self.R,self.D,x);coords=x
  else:expected=prediction(self.L,self.R,self.D,self.mean,self.E,self.B,x);coords=(x-self.mean)@self.E
  actual=prediction_tensor.detach().cpu().double().numpy();cc=code.detach().cpu().double().numpy();active=support.detach().cpu().numpy();assert cc.shape==coords.shape and active.shape==coords.shape and np.all(active==1);pe=float(np.linalg.norm(actual-expected)/max(np.linalg.norm(expected),1e-30));ce=float(np.linalg.norm(cc-coords)/max(np.linalg.norm(coords),1e-30));assert pe<=1e-5 and ce<=1e-5,(pe,ce)
  return pe,ce

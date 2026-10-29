"""Evaluation adapters with explicit executable-input vs privileged-output access."""
import json
import numpy as np
import torch
from bilinear_input_runtime import InputWritePredictor
from fit_pca_encoder import sha,V

class BilinearEvaluationPredictor:
 def __init__(self,entry):
  assert sha(entry['path'])==entry['sha256'];state=torch.load(entry['path'],weights_only=True);self.variant=entry['variant'];self.kind=entry['kind']
  if self.variant=='input':
   initial=next(x for x in json.loads((V/'bilinear_input_initial_states.json').read_text())['models'] if (x['kind'],x['seed'])==(entry['kind'],entry['seed']));assert sha(initial['path'])==initial['sha256'];self.model=InputWritePredictor(torch.load(initial['path'],weights_only=True)).cuda();self.model.load_state_dict(state['state']);self.model.eval()
  elif self.variant=='output_oracle':
   s=state['state'];self.scale=float(s['output_scale']);self.center=s['output_mean'].double().numpy()+self.scale*s['features.bias'].double().numpy();D=s['features.dictionary'].double().numpy();D=D/np.linalg.norm(D,axis=0);u,sv,vt=np.linalg.svd(D,full_matrices=False);keep=sv>sv[0]*1e-10;self.U=u[:,keep];self.inverse=(vt[keep].T/sv[keep])@self.U.T;self.rank=int(keep.sum());self.D=D
  else:raise ValueError(self.variant)
 @torch.no_grad()
 def reconstruct(self,z,write=None):
  if self.variant=='input':
   pred,support,code=self.model(z);return pred,support,code,dict(rank_mean=None,numerical_nonzero_mean=float((code!=0).sum(-1).float().mean()))
  if write is None:raise ValueError('Privileged output oracle requires native write; normalized input is insufficient')
  x=write.detach().cpu().double().numpy()-self.center;pred=(self.center+(x@self.U)@self.U.T).astype('float32');code=(x@self.inverse.T)/self.scale;support=torch.ones(code.shape,device=z.device)
  return torch.from_numpy(pred).to(z.device),support,torch.from_numpy(code).to(z.device),dict(rank_mean=float(self.rank),numerical_nonzero_mean=float((code!=0).sum(-1).mean()))

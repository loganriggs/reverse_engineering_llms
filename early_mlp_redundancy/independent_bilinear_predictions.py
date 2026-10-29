"""Independent NumPy input/privileged-output prediction audit helpers."""
import itertools
import numpy as np
import torch
from fit_pca_encoder import sha
class IndependentBilinearPrediction:
 def __init__(self,entry,tolerances):
  assert sha(entry['path'])==entry['sha256'];s=torch.load(entry['path'],weights_only=True)['state'];self.d={k:v.double().numpy() for k,v in s.items()};self.kind=entry['kind'];self.variant=entry['variant'];self.tol=tolerances;d=self.d;self.D=d['features.dictionary']/np.linalg.norm(d['features.dictionary'],axis=0);self.center=d['output_mean']+d['output_scale']*d['features.bias'];self.M=d['features.membership'];self.ambiguous=0;self.minimum_gap=float('inf')
  if self.variant=='output_oracle':
   self.Q,self.R=np.linalg.qr(self.D,mode='reduced');sv=np.linalg.svd(self.D,compute_uv=False);assert sv[-1]>sv[0]*1e-10,'Rank-deficient oracle requires rank-aware independent audit'
 def verify(self,z,w,out,support,code):
  d=self.d;x=z.double().numpy();actual=out.double().numpy();q=support.numpy();observed_code=code.double().numpy()
  if self.variant=='output_oracle':
   y=w.double().numpy()-self.center;reference=self.center+(y@self.Q)@self.Q.T;cc=np.linalg.solve(self.R,(y.reshape(-1,1152)@self.Q).T).T.reshape(observed_code.shape)/d['output_scale'];assert np.all(q==1)
  else:
   x=(x-d['input_mean'])/d['input_scale'];a=(x@d['features.left'].T+d['features.left_bias'])*(x@d['features.right'].T+d['features.right_bias']);assert np.all((q==0)|(q==1));flat=a.reshape(-1,128);qq=q.reshape(-1,128)
   if self.kind=='dense':assert np.all(q==1)
   else:
    scores=np.abs(flat) if self.kind=='scalar' else flat*flat@self.M.T;k=32 if self.kind=='scalar' else (8 if self.kind=='block' else 4)
    for row,actual_support in zip(scores,qq):
     order=np.argsort(-row,kind='stable');selected=np.zeros_like(row);selected[order[:k]]=1;expected=selected if self.kind=='scalar' else (selected@self.M>0);gap=float(row[order[k-1]]-row[order[k]]);self.minimum_gap=min(self.minimum_gap,gap)
     if np.array_equal(expected,actual_support):continue
     self.ambiguous+=1;cut=row[order[k-1]];eps=self.tol['gate_score_absolute']+self.tol['gate_score_relative']*abs(cut);mandatory=np.flatnonzero(row>cut+eps);eligible=np.flatnonzero(row>=cut-eps)
     if self.kind=='scalar':assert int(actual_support.sum())==k and np.all(actual_support[mandatory]==1) and set(np.flatnonzero(actual_support)).issubset(eligible)
     else:
      required=set(mandatory);optional=[i for i in eligible if i not in required and np.all(self.M[i]<=actual_support)]
      if required:assert np.all((self.M[list(required)].sum(0)>0)<=actual_support)
      found=False
      for extra in itertools.combinations(optional,k-len(required)):
       union=self.M[list(required)+list(extra)].sum(0)>0
       if np.array_equal(union,actual_support):found=True;break
      assert found,'Observed union not attainable inside declared numerical band'
   cc=a*q;reference=(cc@self.D.T)*d['output_scale']+self.center
  pred_error=float(np.linalg.norm(reference-actual)/max(1,np.linalg.norm(reference)));code_error=float(np.linalg.norm(cc-observed_code)/max(1,np.linalg.norm(cc)));assert pred_error<self.tol['output_relative'] and code_error<self.tol['output_relative'],(self.kind,self.variant,pred_error,code_error)
  return pred_error,code_error

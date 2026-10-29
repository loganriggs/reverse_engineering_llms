"""Independent float64 native-product reconstruction with SVD output bases."""
import numpy as np
import torch
from fit_pca_encoder import sha
class IndependentProductPrediction:
 weights={};spaces={}
 def __init__(self,e):
  self.entry=e;self.variant='input';self.kind=e['kind'];self.ix=np.array(e['indices']);self.scale=e['scalar'];key=e['reference_sha256']
  if key not in self.weights:
   assert sha(e['reference_path'])==key;r=torch.load(e['reference_path'],weights_only=True);self.weights[key]=tuple(r[k].double().numpy() for k in ['L','R','D'])
  self.L,self.R,self.W=self.weights[key]
  if e['space']=='native':self.U=None;self.C=self.W[:,self.ix];self.offset=np.zeros(self.W.shape[0])
  else:
   key=e['state_sha256']
   if key not in self.spaces:
    assert sha(e['state_path'])==key;s=torch.load(e['state_path'],weights_only=True)['state'];D=s['features.dictionary'].double().numpy();U,sv,_=np.linalg.svd(D,full_matrices=False);assert min(sv)>max(sv)*1e-10;c=s['output_mean'].double().numpy()+float(s['output_scale'])*s['features.bias'].double().numpy();self.spaces[key]=(U,U.T@self.W,c-U@(U.T@c))
   self.U,C,self.offset=self.spaces[key];self.C=C[:,self.ix]
 def prepare(self,z):
  x=z.detach().cpu().double().numpy();return (x@self.L.T)*(x@self.R.T)
 def verify(self,products,prediction,support,code):
  h=products[...,self.ix];expected=(h@self.C.T)*self.scale
  if self.U is not None:expected=expected@self.U.T
  expected=expected+self.offset
  actual=prediction.detach().cpu().double().numpy();codes=code.detach().cpu().double().numpy();sp=support.detach().cpu().numpy();assert sp.shape==h.shape and np.all(sp==1)
  de=float(np.linalg.norm(expected-actual)/max(np.linalg.norm(expected),1e-30));ce=float(np.linalg.norm(codes-h)/max(np.linalg.norm(h),1e-30));assert de<=1e-5 and ce<=1e-5,(de,ce)
  return de,ce

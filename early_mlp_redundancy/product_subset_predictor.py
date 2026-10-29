"""Executable native-product subsets with separate output dictionaries.

Only normalized input z is consumed. Down_bias is added by the caller.
Support/code count retained native products, not semantic output features.
"""
import json
from pathlib import Path
import numpy as np
import torch
from fit_pca_encoder import sha,V
class ProductSubsetPredictor:
 def __init__(self,entry,device='cuda'):
  self.entry=entry;self.kind=entry['kind'];self.variant='input';self.device=device
  assert sha(entry['reference_path'])==entry['reference_sha256'];r=torch.load(entry['reference_path'],weights_only=True)
  ix=torch.tensor(entry['indices'],dtype=torch.long);assert len(ix)==entry['size'] and len(set(ix.tolist()))==len(ix)
  self.left=r['L'][ix].to(device);self.right=r['R'][ix].to(device);W=r['D'].double().numpy();self.scale=float(entry['scalar'])
  if entry['space']=='native':
   self.down=r['D'][:,ix].to(device);self.basis=None;self.offset=torch.zeros(W.shape[0],device=device);self.output_dimension=W.shape[0]
  else:
   assert sha(entry['state_path'])==entry['state_sha256'];s=torch.load(entry['state_path'],weights_only=True)['state'];D=s['features.dictionary'].double().numpy();Q,_=np.linalg.qr(D,mode='reduced')
   c=s['output_mean'].double().numpy()+float(s['output_scale'])*s['features.bias'].double().numpy()
   self.down=torch.tensor((Q.T@W)[:,ix.numpy()],dtype=torch.float32,device=device);self.basis=torch.tensor(Q,dtype=torch.float32,device=device);self.offset=torch.tensor(c-Q@(Q.T@c),dtype=torch.float32,device=device);self.output_dimension=Q.shape[1]
 @torch.no_grad()
 def reconstruct(self,z,write=None):
  h=torch.nn.functional.linear(z,self.left)*torch.nn.functional.linear(z,self.right)
  y=torch.nn.functional.linear(h,self.down)
  if self.basis is not None:y=torch.nn.functional.linear(y,self.basis)
  y=y*self.scale+self.offset
  return y,torch.ones_like(h),h,{'retained_native_products':h.shape[-1],'output_dimension':self.output_dimension,'rank_mean':None,'numerical_nonzero_mean':float((h!=0).sum(-1).float().mean())}
def intervention_delta(recipient,donor,recipient_positions,donor_positions):
 if len(recipient_positions)!=len(donor_positions):raise ValueError('Position lists must align')
 if len(set(recipient_positions))!=len(recipient_positions):raise ValueError('Repeated recipient position')
 delta=torch.zeros_like(recipient)
 if recipient_positions:
  ix=torch.tensor(recipient_positions,device=recipient.device);di=torch.tensor(donor_positions,device=donor.device)
  delta[:,ix,:]=donor[:,di,:]-recipient[:,ix,:]
 return delta

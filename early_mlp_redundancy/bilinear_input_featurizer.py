"""Executable signed bilinear input features with deterministic union gating."""
import math
import torch
from torch import nn
from torch.nn import functional as F
KINDS=('dense','scalar','block','shared_fixed')
class BilinearInputFeaturizer(nn.Module):
 def __init__(self,kind,seed):
  super().__init__()
  if kind not in KINDS:raise ValueError(kind)
  self.kind=kind;g=torch.Generator().manual_seed(seed)
  self.left=nn.Parameter(torch.randn(128,1152,generator=g)/math.sqrt(1152))
  self.right=nn.Parameter(torch.randn(128,1152,generator=g)/math.sqrt(1152))
  self.dictionary=nn.Parameter(F.normalize(torch.randn(1152,128,generator=g),dim=0))
  self.left_bias=nn.Parameter(torch.zeros(128));self.right_bias=nn.Parameter(torch.zeros(128));self.bias=nn.Parameter(torch.zeros(1152))
  membership=torch.zeros(32,128)
  if kind=='block':membership[torch.arange(32)[:,None],torch.arange(128).reshape(32,4)]=1
  elif kind=='shared_fixed':
   mg=torch.Generator().manual_seed(seed+10000)
   for half in range(2):membership[torch.arange(16)[:,None]+16*half,torch.randperm(128,generator=mg).reshape(16,8)]=1
  self.register_buffer('membership',membership)
 def select(self,a):
  if self.kind=='dense':return torch.ones_like(a),None
  score=a.abs() if self.kind=='scalar' else a.square()@self.membership.T
  k=32 if self.kind=='scalar' else (8 if self.kind=='block' else 4)
  ids=torch.argsort(score,dim=-1,descending=True,stable=True)[...,:k]
  selected=torch.zeros_like(score).scatter_(-1,ids,1)
  return (selected if self.kind=='scalar' else (selected@self.membership>0).to(a.dtype)),selected
 def forward(self,x):
  a=F.linear(x,self.left,self.left_bias)*F.linear(x,self.right,self.right_bias)
  support,_=self.select(a);code=a*support
  return code@F.normalize(self.dictionary,dim=0).T+self.bias,support,code

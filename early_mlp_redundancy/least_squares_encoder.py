"""M72 independent per-input least-squares encoder with frozen decoder/supports."""
from collections import OrderedDict
import numpy as np,torch
from torch.nn import functional as F
class LeastSquaresEncoder:
 def __init__(self,f,mean,scale):
  self.f=f;self.mean=mean;self.scale=scale;self.mu=mean.detach().cpu().double().numpy();self.s=float(scale);self.D=F.normalize(f.dictionary.detach(),dim=0).cpu().double().numpy();self.bias=f.bias.detach().cpu().double().numpy();self.cache=OrderedDict()
 def solver(self,inds):
  key=tuple(inds)
  if key in self.cache:self.cache.move_to_end(key);return self.cache[key]
  D=self.D[:,inds]
  if len(inds):u,sv,vt=np.linalg.svd(D,full_matrices=False);keep=sv>1e-10*sv[0];pinv=(vt[keep].T/sv[keep])@u[:,keep].T;rank=int(keep.sum())
  else:pinv=np.zeros((0,1152));rank=0
  self.cache[key]=(pinv,rank)
  if len(self.cache)>256:self.cache.popitem(last=False)
  return pinv,rank
 @torch.no_grad()
 def reconstruct(self,write,variant):
  pred,q,original_code=self.f((write-self.mean)/self.scale)
  if variant=='original':return pred*self.scale+self.mean,q,original_code,dict(rank=None,numerical_nonzero_mean=float((original_code!=0).sum(-1).float().mean()))
  raw=write.detach().cpu().double().numpy();shape=raw.shape;target=((raw-self.mu)/self.s-self.bias).reshape(-1,1152);support=q.detach().cpu().numpy().reshape(-1,128)>0 if variant=='own_support_LS' else np.ones((len(target),128),dtype=bool);assert variant in ['own_support_LS','full_dictionary_LS'];codes=np.zeros((len(target),128));ranks=np.zeros(len(target),dtype=int);groups={}
  for i,row in enumerate(support):groups.setdefault(tuple(np.flatnonzero(row)),[]).append(i)
  for inds,idx in groups.items():pinv,rank=self.solver(inds);codes[np.ix_(idx,list(inds))]=target[idx]@pinv.T;ranks[idx]=rank
  rec=(self.mu+self.s*(codes@self.D.T+self.bias)).reshape(shape).astype(np.float32);code=codes.reshape(shape[:-1]+(128,));qout=torch.from_numpy(support.reshape(code.shape).astype(np.float32)).to(write.device)
  return torch.from_numpy(rec).to(write.device),qout,torch.from_numpy(code).to(write.device),dict(rank=ranks.reshape(shape[:-1]).tolist(),rank_mean=float(ranks.mean()),numerical_nonzero_mean=float((codes!=0).sum(-1).mean()))

import torch
class BatchedCoverage(torch.nn.Module):
 def __init__(self,A,B,W):
  super().__init__();self.A=torch.nn.Parameter(A.clone());self.B=torch.nn.Parameter(B.clone());self.W=torch.nn.Parameter(W.clone());self.i,self.j=torch.triu_indices(A.shape[1],A.shape[1],device=A.device)
 def forward(self,x):
  z=torch.einsum('nd,bhd->bnh',x,self.A)*torch.einsum('nd,bhd->bnh',x,self.B)
  return torch.einsum('bnp,btp->bnt',z[:,:,self.i]*z[:,:,self.j],self.W)
def losses(model,x,y,mask):
 return (((model(x)-y[None])**2)*mask).sum(dim=(1,2))/mask.sum(dim=(1,2))
def clip_each(model,limit=10.):
 norm=torch.sqrt(sum(p.grad.square().flatten(1).sum(dim=1) for p in model.parameters()));scale=(limit/norm.clamp_min(1e-30)).clamp(max=1.)
 for p in model.parameters():p.grad.mul_(scale.reshape(-1,*([1]*(p.ndim-1))))
 return norm

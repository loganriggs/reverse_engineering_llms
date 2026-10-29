"""Polynomial depth2–5 shared/separate trunks; fixed calibration scales."""
import math
import torch
from torch import nn


class DeepMultiTask(nn.Module):
    def __init__(self,d,h,tasks,depth,shared,seed):
        super().__init__()
        assert depth>=2
        self.d,self.h,self.tasks,self.depth,self.shared=d,h,tasks,depth,shared
        self.A=nn.ParameterList();self.B=nn.ParameterList()
        g=torch.Generator().manual_seed(seed)
        for layer in range(depth-1):
            width=d if layer==0 else h
            a=torch.randn(h,width,generator=g,dtype=torch.float64)/math.sqrt(width)
            b=torch.randn(h,width,generator=g,dtype=torch.float64)/math.sqrt(width)
            self.A.append(nn.Parameter(a if shared else a[None].repeat(tasks,1,1)))
            self.B.append(nn.Parameter(b if shared else b[None].repeat(tasks,1,1)))
            self.register_buffer('scale_'+str(layer),torch.ones(h if shared else (tasks,h),dtype=torch.float64))
        self.W=nn.Parameter(torch.randn(tasks,h*(h+1)//2,generator=g,dtype=torch.float64)/math.sqrt(h*(h+1)//2))
        i,j=torch.triu_indices(h,h);self.register_buffer('i',i);self.register_buffer('j',j)

    def hidden(self,x):
        z=x if self.shared else x[:,None,:].expand(-1,self.tasks,-1)
        for l,(a,b) in enumerate(zip(self.A,self.B)):
            z=((z@a.T)*(z@b.T) if self.shared else
               torch.einsum('ntd,thd->nth',z,a)*torch.einsum('ntd,thd->nth',z,b))
            z=z/getattr(self,'scale_'+str(l))
        return z

    def forward(self,x,recompute=False):
        if recompute and self.shared:
            return torch.stack([(self.hidden(x)[:,self.i]*self.hidden(x)[:,self.j])@w for w in self.W],dim=1)
        z=self.hidden(x)
        if self.shared:return (z[:,self.i]*z[:,self.j])@self.W.T
        return torch.einsum('ntp,tp->nt',z[:,:,self.i]*z[:,:,self.j],self.W)

    @torch.no_grad()
    def calibrate(self,x):
        """One-time initialization, using independent calibration inputs only.

        Subsequent scales are fixed: this is still a homogeneous polynomial,
        not input-dependent layer normalization.
        """
        z=x if self.shared else x[:,None,:].expand(-1,self.tasks,-1)
        for l,(a,b) in enumerate(zip(self.A,self.B)):
            z=((z@a.T)*(z@b.T) if self.shared else
               torch.einsum('ntd,thd->nth',z,a)*torch.einsum('ntd,thd->nth',z,b))
            scale=z.square().mean(dim=0).sqrt().clamp_min(1e-12)
            getattr(self,'scale_'+str(l)).copy_(scale);z=z/scale
        rms=self(x).square().mean(dim=0).sqrt().clamp_min(1e-12)
        self.W.div_(rms[:,None])


def independent_teacher(d,h,tasks,depth,seed):
    """Distinct task trunks, unlike matched student initialization."""
    model=DeepMultiTask(d,h,tasks,depth,False,seed)
    with torch.no_grad():
        for t in range(tasks):
            single=DeepMultiTask(d,h,1,depth,True,seed+100*(t+1))
            for dest,src in zip(model.A,single.A):dest[t].copy_(src)
            for dest,src in zip(model.B,single.B):dest[t].copy_(src)
            model.W[t].copy_(single.W[0])
    return model

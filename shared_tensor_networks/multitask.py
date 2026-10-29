"""Multi-task quartic models with shared or separate quadratic dictionaries."""
import itertools
import math
import numpy as np
import torch
from torch import nn

from experiment import multiply


def monomials(d):
    return sorted({tuple(sum(i==j for i in inds) for j in range(d)) for inds in itertools.combinations_with_replacement(range(d),4)})


def product_map(A,B,keys):
    d=A.shape[1];unit=[tuple(int(i==j) for i in range(d)) for j in range(d)]
    quadratic=[multiply(dict(zip(unit,a)),dict(zip(unit,b))) for a,b in zip(A,B)]
    products=[multiply(quadratic[i],quadratic[j]) for i,j in itertools.combinations_with_replacement(range(len(A)),2)]
    return np.array([[p.get(k,0.) for p in products] for k in keys])


class MultiTask(nn.Module):
    def __init__(self,d,h,tasks,shared,seed):
        super().__init__();self.shared=shared;self.d=d;self.h=h;self.tasks=tasks
        rng=torch.Generator().manual_seed(seed)
        a=torch.randn(h,d,generator=rng)/math.sqrt(d);b=torch.randn(h,d,generator=rng)/math.sqrt(d)
        # Equal functions initially for the width-matched shared/separate pair.
        self.A=nn.Parameter(a if shared else a[None].repeat(tasks,1,1))
        self.B=nn.Parameter(b if shared else b[None].repeat(tasks,1,1))
        self.W=nn.Parameter(torch.randn(tasks,h*(h+1)//2,generator=rng)/math.sqrt(h*(h+1)//2))
        pairs=torch.triu_indices(h,h);self.register_buffer('i',pairs[0]);self.register_buffer('j',pairs[1])

    def forward(self,x,recompute=False):
        if self.shared:
            if recompute:
                outputs=[]
                for t in range(self.tasks):
                    z=(x@self.A.T)*(x@self.B.T);p=z[:,self.i]*z[:,self.j];outputs.append(p@self.W[t])
                return torch.stack(outputs,dim=1)
            z=(x@self.A.T)*(x@self.B.T);p=z[:,self.i]*z[:,self.j]
            return p@self.W.T
        z=torch.einsum('nd,thd->nth',x,self.A)*torch.einsum('nd,thd->nth',x,self.B)
        p=z[:,:,self.i]*z[:,:,self.j]
        return torch.einsum('ntp,tp->nt',p,self.W)

    def coefficient_matrix(self,keys):
        a=self.A.detach().numpy();b=self.B.detach().numpy();w=self.W.detach().numpy()
        if self.shared:return product_map(a,b,keys)@w.T
        return np.column_stack([product_map(a[t],b[t],keys)@w[t] for t in range(self.tasks)])


def make_teacher(d,h,tasks,shared,seed,keys,H):
    rng=np.random.default_rng(seed);count=1 if shared else tasks
    A=rng.normal(size=(count,h,d))/math.sqrt(d);B=rng.normal(size=(count,h,d))/math.sqrt(d)
    W=rng.normal(size=(tasks,h*(h+1)//2))/math.sqrt(h*(h+1)//2)
    K=[product_map(A[0 if shared else t],B[0 if shared else t],keys) for t in range(tasks)]
    C=np.column_stack([k@w for k,w in zip(K,W)])
    norms=np.sqrt(np.sum(C*(H@C),axis=0));C/=norms;W/=norms[:,None]
    return C,dict(A=A,B=B,W=W)

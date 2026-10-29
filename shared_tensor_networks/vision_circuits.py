"""Bilinear stack versus its fully untied binary occurrence tree."""
import math
import torch
from torch import nn


class BilinearCircuit(nn.Module):
    def __init__(self,d,h,depth,classes=10,tree=False,normalize=False,seed=0,dtype=torch.float32):
        super().__init__();self.d=d;self.h=h;self.depth=depth;self.tree=tree;self.normalize=normalize
        assert depth>=1
        rng=torch.Generator().manual_seed(seed)
        # Same level tensors are copied into distinct tree occurrences.
        pairs=[(torch.randn(h,d if l==0 else h,generator=rng,dtype=dtype)/math.sqrt(d if l==0 else h),
                torch.randn(h,d if l==0 else h,generator=rng,dtype=dtype)/math.sqrt(d if l==0 else h)) for l in range(depth)]
        self.levels=([depth-1-int(math.log2(i+1)) for i in range(2**depth-1)] if tree else list(range(depth)))
        self.A=nn.ParameterList([nn.Parameter(pairs[l][0].clone()) for l in self.levels])
        self.B=nn.ParameterList([nn.Parameter(pairs[l][1].clone()) for l in self.levels])
        self.W=nn.Parameter(torch.randn(classes,h,generator=rng,dtype=dtype)/math.sqrt(h))

    def combine(self,left,right,i):
        z=(left@self.A[i].T)*(right@self.B[i].T)
        if self.normalize:z=z/torch.sqrt(z.square().mean(dim=-1,keepdim=True)+1e-6)
        return z

    def forward(self,x):
        if self.tree:
            def visit(i):
                if self.levels[i]==0:return self.combine(x,x,i)
                return self.combine(visit(2*i+1),visit(2*i+2),i)
            h=visit(0)
        else:
            h=x
            for i in range(self.depth):h=self.combine(h,h,i)
        return h@self.W.T


def parameter_count(d,h,depth,classes=10,tree=False):
    if tree:return 2**depth*d*h+2*(2**(depth-1)-1)*h*h+classes*h
    return 2*d*h+2*(depth-1)*h*h+classes*h


def nearest_tree_width(d,h,depth,classes=10):
    target=parameter_count(d,h,depth,classes)
    return min(range(1,h+1),key=lambda w:abs(parameter_count(d,w,depth,classes,True)-target))

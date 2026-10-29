"""Independent occurrence tree with a separately specified width at each level."""
import math
import torch
from torch import nn
from vision_circuits import BilinearCircuit,parameter_count

class NonuniformTree(BilinearCircuit):
    def __init__(self,d,widths,classes=10,normalize=True,seed=0,dtype=torch.float32):
        nn.Module.__init__(self);self.d=d;self.widths=list(widths);self.depth=len(widths);self.tree=True;self.normalize=normalize
        assert self.depth and all(w>0 for w in widths)
        rng=torch.Generator().manual_seed(seed);pairs=[]
        for level,width in enumerate(widths):
            incoming=d if level==0 else widths[level-1]
            pairs.append((torch.randn(width,incoming,generator=rng,dtype=dtype)/math.sqrt(incoming),torch.randn(width,incoming,generator=rng,dtype=dtype)/math.sqrt(incoming)))
        self.levels=[self.depth-1-int(math.log2(i+1)) for i in range(2**self.depth-1)]
        self.A=nn.ParameterList([nn.Parameter(pairs[l][0].clone()) for l in self.levels]);self.B=nn.ParameterList([nn.Parameter(pairs[l][1].clone()) for l in self.levels]);self.W=nn.Parameter(torch.randn(classes,widths[-1],generator=rng,dtype=dtype)/math.sqrt(widths[-1]))

def count(d,widths,classes=10):
    depth=len(widths)
    return sum(2*2**(depth-1-l)*w*(d if l==0 else widths[l-1]) for l,w in enumerate(widths))+classes*widths[-1]

def budget_widths(d,top,depth,classes=10):
    target=parameter_count(d,top,depth,classes)
    leaf=min(range(1,top+1),key=lambda b:abs(count(d,[b]+[top]*(depth-1),classes)-target))
    return [leaf]+[top]*(depth-1)

"""Exact Gaussian inner products of arbitrary products of four linear forms.

Supports different dictionaries and within-head unions without inventing
cross-head product features. Blocked implementation bounds temporary memory.
"""
import numpy as np
from exact_scaling_task_span import pairings
PAIRINGS=tuple(pairings(tuple(range(8))))

def cross_gram(left,right,block=64):
    left=np.asarray(left,dtype=float);right=np.asarray(right,dtype=float)
    if left.ndim!=3 or right.ndim!=3 or left.shape[1]!=4 or right.shape[1]!=4 or left.shape[2]!=right.shape[2]:
        raise ValueError('expected (features,4,input_dimension) arrays')
    if block<1:raise ValueError('positive block size required')
    out=np.empty((len(left),len(right)))
    for lo in range(0,len(left),block):
      L=left[lo:lo+block]
      for ro in range(0,len(right),block):
        R=right[ro:ro+block];cov={}
        for a in range(8):
          for b in range(a+1,8):
            if b<4:cov[a,b]=np.sum(L[:,a]*L[:,b],axis=1)[:,None]
            elif a>=4:cov[a,b]=np.sum(R[:,a-4]*R[:,b-4],axis=1)[None,:]
            else:cov[a,b]=L[:,a]@R[:,b-4].T
        total=np.zeros((len(L),len(R)))
        for pairing in PAIRINGS:
          value=np.ones_like(total)
          for pair in pairing:value*=cov[pair]
          total+=value
        out[lo:lo+len(L),ro:ro+len(R)]=total
    return out

def product_factors(A,B,scale):
    """One trunk's unordered products; concatenate results for separate heads."""
    i,j=np.triu_indices(len(A))
    return np.stack([A[i]/scale[i,None],B[i],A[j]/scale[j,None],B[j]],axis=1)

def sphere_radial_moment(d):
    return d*(d+2)*(d+4)*(d+6)

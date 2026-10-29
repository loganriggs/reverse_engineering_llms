"""Continuous, positively homogeneous block shrinkage with at most k active groups.

Input shape [..., groups, coordinates_per_group]. No learned parameters.
The threshold is the (k+1)th largest group norm, with gradients retained.
At threshold ties, groups exactly at the threshold vanish; derivatives may
be nonunique, but the output is continuous. Zero-norm groups return zero.
"""
import torch


def block_shrinkage(a,k):
    if a.ndim<2 or not a.is_floating_point():raise ValueError('floating tensor with group and coordinate axes required')
    groups=a.shape[-2]
    if not isinstance(k,int) or not 0<=k<=groups:raise ValueError('k must be an integer in [0, groups]')
    if k==0:return a*0
    if k==groups:return a
    norms=torch.linalg.vector_norm(a,dim=-1)
    threshold=norms.topk(k+1,dim=-1).values[...,-1:]
    denominator=torch.where(norms>0,norms,torch.ones_like(norms))
    multiplier=torch.relu(norms-threshold)/denominator
    return a*multiplier.unsqueeze(-1)

"""Deterministic balanced grouping, fit exclusively on caller-supplied training data."""
import torch

def balanced_four(features):
 features=features.double();features=features/features.norm(dim=1,keepdim=True).clamp_min(1e-30);similarity=features@features.T;clusters=[[i] for i in range(len(features))]
 assert len(clusters)==128
 for _ in range(2):
  C=torch.zeros(len(clusters),128,dtype=torch.double)
  for i,c in enumerate(clusters):C[i,c]=1/len(c)
  scores=C@similarity@C.T;mask=torch.triu(torch.ones_like(scores,dtype=torch.bool),diagonal=1);scores[~mask]=-float('inf');new=[]
  while len(new)<len(clusters)//2:
   flat=int(scores.argmax());a,b=divmod(flat,len(clusters));assert torch.isfinite(scores[a,b]);new.append(clusters[a]+clusters[b]);scores[a,:]=-float('inf');scores[:,a]=-float('inf');scores[b,:]=-float('inf');scores[:,b]=-float('inf')
  clusters=new
 M=torch.zeros(32,128,dtype=torch.double)
 for i,c in enumerate(clusters):M[i,c]=1
 assert bool((M.sum(0)==1).all()) and bool((M.sum(1)==4).all())
 return M

def pair_jaccard(A,B):
 mask=torch.triu(torch.ones(128,128,dtype=torch.bool),diagonal=1);a=(A.T@A>0)&mask;b=(B.T@B>0)&mask
 return float((a&b).sum()/(a|b).sum())

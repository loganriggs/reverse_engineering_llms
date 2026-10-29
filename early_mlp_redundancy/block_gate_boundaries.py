"""Exact candidate crossings of TopK block energies along an affine input path."""
import numpy as np


def support(a,b,t,k):
    energy=np.square(a+t*b).sum(axis=1)
    order=np.argsort(-energy,kind='stable')
    return tuple(sorted(order[:k].tolist()))


def boundaries(a,b,k):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    assert a.shape==b.shape and a.ndim==2 and 0<k<len(a)
    poly=np.stack([np.square(a).sum(1),2*(a*b).sum(1),np.square(b).sum(1)],axis=1)
    roots=[];permanent_ties=[]
    for i in range(len(a)):
      for j in range(i):
        c=poly[i]-poly[j]
        if np.all(c==0):permanent_ties.append([j,i]);continue
        for root in np.polynomial.polynomial.polyroots(c):
          if abs(root.imag)<1e-10 and 0<float(root.real)<1:roots.append(float(root.real))
    roots=sorted(roots);unique=[]
    for t in roots:
        if not unique or t-unique[-1]>1e-10:unique.append(t)
    knots=[0.]+unique+[1.]
    intervals=[support(a,b,(l+r)/2,k) for l,r in zip(knots[:-1],knots[1:])]
    changes=[]
    for i,t in enumerate(unique):
      if intervals[i]!=intervals[i+1]:
        changes.append(dict(t=t,left=list(intervals[i]),right=list(intervals[i+1]),radius=min(t-knots[i],knots[i+2]-t)/2))
    return changes,permanent_ties


def decoded(a,b,t,D,selected):
    code=np.zeros_like(a);code[selected]=(a+t*b)[selected]
    return D@code.reshape(-1)

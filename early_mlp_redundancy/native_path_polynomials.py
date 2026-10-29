"""Native bilinear computation on a normalized chord: rational quadratic codes, quartic ties."""
import numpy as np


def path_coefficients(z0,z1,L,R,D,E,mu,scale,bias,radius):
    z0=z0/np.linalg.norm(z0)*radius;z1=z1/np.linalg.norm(z1)*radius;d=z1-z0
    q=np.array([z0@z0,2*(z0@d),d@d]);left=L@z0;ld=L@d;right=R@z0;rd=R@d
    w=radius**2*np.stack([D@(left*right),D@(left*rd+ld*right),D@(ld*rd)])
    constant=bias-E@mu/scale
    numerator=w@E.T/scale+q[:,None]*constant
    return z0,z1,q,w,numerator


def value(coeff,t):return np.polynomial.polynomial.polyval(t,coeff)


def energies(numerator):
    # [3, groups, blockdim] -> [groups,5]
    a,b,c=numerator
    return np.stack([(a*a).sum(-1),2*(a*b).sum(-1),(b*b+2*a*c).sum(-1),2*(b*c).sum(-1),(c*c).sum(-1)],-1)


def select(numerator,t,k):
    a=value(numerator,t);return tuple(sorted(np.argsort(-np.square(a).sum(-1),kind='stable')[:k].tolist()))


def crossing_intervals(numerator,k):
    energy=energies(numerator);roots=[];ties=[]
    for i in range(len(energy)):
      for j in range(i):
        poly=energy[i]-energy[j];norm=np.max(np.abs(poly))
        if norm==0:ties.append([j,i]);continue
        for root in np.polynomial.polynomial.polyroots(poly/norm):
          if abs(root.imag)<1e-8 and 0<float(root.real)<1:roots.append(float(root.real))
    roots.sort();knots=[0.]
    for t in roots:
        if t-knots[-1]>1e-9:knots.append(t)
    knots.append(1.);ss=[select(numerator,(a+b)/2,k) for a,b in zip(knots[:-1],knots[1:])]
    changes=[]
    for i in range(1,len(knots)-1):
      if ss[i-1]!=ss[i]:changes.append(dict(t=knots[i],left=list(ss[i-1]),right=list(ss[i]),radius=min(knots[i]-knots[i-1],knots[i+1]-knots[i])/2))
    return changes,ties

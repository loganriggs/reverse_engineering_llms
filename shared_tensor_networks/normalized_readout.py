"""Unit-factor bilinear features with explicit radial penalty.

Prediction uses unit A/B rows. Residual appends sqrt(gamma)*(raw_row_norm-1),
which fixes irrelevant raw scales without restricting factor directions.
"""
import numpy as np
from direct_bilinear_readout import predict_jacobian


def normalized_design(theta,x,h):
    d=x.shape[1];rows=theta.reshape(2*h,d);norm=np.linalg.norm(rows,axis=1)
    if np.any(norm<1e-12):raise ValueError('zero factor outside normalized chart')
    unit=rows/norm[:,None];k=len(theta);p=h*(h+1)//2
    phi,J=predict_jacobian(np.r_[unit.ravel(),np.eye(p).ravel()],x,h)
    chain=np.zeros((k,k));N=np.zeros((2*h,k))
    for i in range(2*h):
        sl=slice(i*d,(i+1)*d)
        chain[sl,sl]=(np.eye(d)-np.outer(unit[i],unit[i]))/norm[i]
        N[i,sl]=unit[i]
    return phi,J[:,:,:k]@chain,norm-1,N,unit


def objective(v,x,y,h,ridge=.001,gamma=1.,reduced=True):
    k=2*h*x.shape[1];p=h*(h+1)//2;t=y.shape[1]
    phi,D,norm,N,unit=normalized_design(v[:k],x,h)
    Z=np.concatenate([phi,np.sqrt(ridge)*np.eye(p)])
    Y=np.concatenate([y,np.zeros((p,t))])
    if reduced:
        U,_,rank,_=np.linalg.lstsq(Z,Y,rcond=None)
        if rank!=p:raise ValueError('numerically rank deficient augmented design')
    else:U=v[k:].reshape(t,p).T
    r=Z@U-Y
    J=np.empty((*r.shape,len(v)))
    if reduced:_,R=np.linalg.qr(Z,mode='reduced')
    for j in range(k):
        dz=np.concatenate([D[:,:,j],np.zeros((p,p))]);base=dz@U
        if reduced:
            rhs=-dz.T@r-Z.T@base
            du=np.linalg.solve(R,np.linalg.solve(R.T,rhs));base+=Z@du
        J[:,:,j]=base
    if not reduced:
        J[:,:,k:]=0
        for i in range(t):J[:,i,k+i*p:k+(i+1)*p]=Z
    radial=np.zeros((2*h,len(v)));radial[:,:k]=np.sqrt(gamma)*N
    return np.r_[r.ravel(),np.sqrt(gamma)*norm],np.concatenate([J.reshape(-1,len(v)),radial]),U,unit

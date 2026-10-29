import numpy as np
from normalized_readout import objective as normalized_objective
from variable_projection import reduced_residual
from direct_bilinear_readout import predict_jacobian


def objective(v,x,y,h,ridge,normalized,reduced):
    if normalized:return normalized_objective(v,x,y,h,ridge=ridge,reduced=reduced)
    k=2*h*x.shape[1];p=h*(h+1)//2;t=y.shape[1]
    if reduced:
        r,J,U=reduced_residual(v,x,y,h,ridge)
    else:
        pred,J0=predict_jacobian(v,x,h);U=v[k:].reshape(t,p).T
        r=np.concatenate([pred-y,np.sqrt(ridge)*U])
        J=np.zeros((*r.shape,len(v)));J[:len(x)]=J0
        for i in range(t):J[len(x):,i,k+i*p:k+(i+1)*p]=np.sqrt(ridge)*np.eye(p)
    return r.ravel(),J.reshape(-1,len(v)),U,v[:k].reshape(2*h,x.shape[1])

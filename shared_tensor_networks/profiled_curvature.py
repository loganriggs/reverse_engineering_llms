import numpy as np
from normalized_readout import objective
from direct_bilinear_readout import predict_jacobian

def evaluate(theta,x,y,h=4,derivatives=True):
    rows=theta.reshape(2*h,x.shape[1]);norm=np.linalg.norm(rows,axis=1)
    if np.any(norm<1e-12):raise ValueError('zero factor')
    ab=rows/norm[:,None];p=h*(h+1)//2
    phi=predict_jacobian(np.r_[ab.ravel(),np.eye(p).ravel()],x,h,False)
    Z=np.concatenate([phi,np.sqrt(.001)*np.eye(p)]);Y=np.concatenate([y,np.zeros((p,y.shape[1]))])
    if not derivatives:
        U,_,rank,_=np.linalg.lstsq(Z,Y,rcond=None)
        if rank!=p:raise ValueError('readout rank')
        return np.r_[(Z@U-Y).ravel(),norm-1],U,ab
    r,J,U,ab=objective(theta,x,y,h)
    Q,_=np.linalg.qr(Z,mode='reduced')
    tensor=J[:-2*h].reshape(len(Z),y.shape[1],len(theta))
    B=np.einsum('np,ptk->ntk',Q,np.einsum('np,ntk->ptk',Q,tensor))
    A=J.copy();A[:-2*h]-=B.reshape(-1,len(theta))
    return r,J,A,U,ab

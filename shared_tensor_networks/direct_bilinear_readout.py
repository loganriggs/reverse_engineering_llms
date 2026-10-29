"""Direct sample-space predictions/Jacobian for shared quadratic product features."""
import itertools
import numpy as np

def predict_jacobian(v,x,h,need_jacobian=True):
 d=x.shape[1];n=h*d;p=h*(h+1)//2
 A=v[:n].reshape(h,d);B=v[n:2*n].reshape(h,d);W=v[2*n:].reshape(-1,p)
 ax,bx=x@A.T,x@B.T;z=ax*bx;pairs=list(itertools.combinations_with_replacement(range(h),2))
 products=np.stack([z[:,i]*z[:,j] for i,j in pairs],axis=1);pred=products@W.T
 if not need_jacobian:return pred
 dz=np.zeros((len(x),len(W),h))
 for k,(i,j) in enumerate(pairs):
  dz[:,:,i]+=z[:,j,None]*W[None,:,k];dz[:,:,j]+=z[:,i,None]*W[None,:,k]
 J=np.zeros((len(x),len(W),len(v)))
 J[:,:,:n]=(dz[:,:,:,None]*bx[:,None,:,None]*x[:,None,None,:]).reshape(len(x),len(W),n)
 J[:,:,n:2*n]=(dz[:,:,:,None]*ax[:,None,:,None]*x[:,None,None,:]).reshape(len(x),len(W),n)
 for t in range(len(W)):J[:,t,2*n+t*p:2*n+(t+1)*p]=products
 return pred,J

def quadratic_coefficients(A,B):
 d=A.shape[1]
 return np.stack([A[:,i]*B[:,j]+(A[:,j]*B[:,i] if i!=j else 0) for i,j in itertools.combinations_with_replacement(range(d),2)])

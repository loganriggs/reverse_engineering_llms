"""Exact bilinear-Jacobian context metric and metric PCA; NumPy float64 reference."""
import numpy as np

def jacobian(L,R,D,z):
 return (D*(R@z)[None,:])@L+(D*(L@z)[None,:])@R

def operator_metric(L,R,D,S=None):
 """Q satisfies dz.T Q dz = tr(J(dz) S J(dz).T)."""
 S=np.eye(L.shape[1]) if S is None else S
 H=D.T@D;LL=L@S@L.T;RR=R@S@R.T;LR=L@S@R.T
 cross=R.T@(H*LR)@L
 Q=R.T@(H*LL)@R+L.T@(H*RR)@L+cross+cross.T
 return (Q+Q.T)/2

def fit_metric_pca(X,Q,k,relative_cutoff=1e-10):
 mean=X.mean(0);w,U=np.linalg.eigh(Q);scale=max(float(w.max()),1e-300)
 if w.min() < -1e-10*scale:raise ValueError('Metric is not positive semidefinite')
 keep=w>relative_cutoff*scale;Qhalf=(U[:,keep]*np.sqrt(w[keep]))@U[:,keep].T;Qinvhalf=(U[:,keep]/np.sqrt(w[keep]))@U[:,keep].T
 Y=(X-mean)@Qhalf;C=Y.T@Y/len(X);e,V=np.linalg.eigh((C+C.T)/2);order=np.argsort(e)[::-1];V=V[:,order[:k]]
 return dict(mean=mean,encoder=Qhalf@V,decoder=Qinvhalf@V,eigenvalues=e[order],metric_rank=int(keep.sum()))

def reconstruct(model,X):
 return model['mean']+((X-model['mean'])@model['encoder'])@model['decoder'].T

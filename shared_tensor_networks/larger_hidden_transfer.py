import numpy as np
from quartic_cross_gram import cross_gram,product_factors,sphere_radial_moment

def dictionary(state,heads=None):
    A=state['A.0'].numpy();B=state['B.0'].numpy();sc=state['scale_0'].numpy();W=state['W'].numpy()
    if heads is not None:W=W[heads]
    if A.ndim==2:
        F=product_factors(A,B,sc);return F,W.T
    ids=list(range(len(A))) if heads is None else list(heads);p=W.shape[1]
    F=np.concatenate([product_factors(A[t],B[t],sc[t]) for t in ids]);T=np.zeros((len(F),len(ids)))
    for j in range(len(ids)):T[j*p:(j+1)*p,j]=W[j]
    return F,T

def values(F,x):return np.prod(np.einsum('nd,pkd->npk',x,F),axis=2)
def sphere_factor(d):return d**4/sphere_radial_moment(d)
def ridge(Z,Y):
    n,p=Z.shape
    if p>n:return Z.T@np.linalg.solve(Z@Z.T+n*.001*np.eye(n),Y)
    return np.linalg.solve(Z.T@Z+n*.001*np.eye(p),Z.T@Y)
def target_moments(F,teacher,d):
    factor=sphere_factor(d);energy=[];cross=[]
    for t in range(12,16):
        Ft,T=dictionary(teacher,[t]);w=T[:,0]
        energy.append(float(w@cross_gram(Ft,Ft)@w*factor))
        cross.append(cross_gram(F,Ft)@w*factor)
    return np.column_stack(cross),np.array(energy)

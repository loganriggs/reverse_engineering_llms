"""Bounded-sample evaluation helpers for arbitrary hidden width."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from quartic_cross_gram import product_factors,cross_gram

def features(A,B,x):
    z=(x@A.T)*(x@B.T);i,j=np.triu_indices(len(A));return z[:,i]*z[:,j]

def empirical_mse(A,B,W,x,y,block_size=256):
    total=0.
    for start in range(0,len(x),block_size):
        error=features(A,B,x[start:start+block_size])@W.T-y[start:start+block_size]
        total+=float(np.sum(error*error))
    return total/y.size

def components(A,B,TA,TB):
    def matrix(a,b):return (a[:,:,None]*b[:,None,:]+b[:,:,None]*a[:,None,:])/2
    def inner(a,b):return 2*np.einsum('iab,jab->ij',a,b)+np.outer(np.trace(a,axis1=1,axis2=2),np.trace(b,axis1=1,axis2=2))
    a,b=matrix(A,B),matrix(TA,TB);numerator=abs(inner(a,b));denominator=np.sqrt(np.outer(np.diag(inner(a,a)),np.diag(inner(b,b))))
    C=np.divide(numerator,denominator,out=np.zeros_like(numerator),where=denominator>0)
    i,j=linear_sum_assignment(-C);return C[i,j]

def teacher_geometry(A,B,W):
    F=product_factors(A,B,np.ones(len(A)));G=cross_gram(F,F)
    return dict(F=F,G=G,energy=np.sum((W@G)*W,axis=1))

def candidate_geometry(A,B,teacher):
    F=product_factors(A,B,np.ones(len(A)))
    return dict(F=F,G=cross_gram(F,F),C=cross_gram(F,teacher['F']))

def population_errors(U,G,C,TW,energy):
    """Readout U is feature×task; TW is task×teacher_feature."""
    return np.sum(U*(G@U),axis=0)-2*np.sum(U*(C@TW.T),axis=0)+energy

def adaptation(A,B,x,y,G,C,TW,energy,budget):
    X=features(A,B,x[:budget]);coef,_,rank,s=np.linalg.lstsq(X,y[:budget],rcond=1e-12)
    errors=population_errors(coef,G,C,TW,energy)
    return coef,dict(errors=errors.tolist(),rank=int(rank),largest_singular=float(s[0]),smallest_retained=float(s[rank-1]) if rank else None)

"""Unit-row gradient/retraction and exact frozen-feature readout flow."""
import itertools
import numpy as np

PAIRS=list(itertools.combinations_with_replacement(range(4),2))
RIDGE=.001

def features(rows,x):
    a=x@rows[:4].T;b=x@rows[4:].T;z=a*b
    return np.stack([z[:,i]*z[:,j] for i,j in PAIRS],1),a,b,z

def calc(rows,U,x,y,gradient=True):
    f,a,b,z=features(rows,x);r=f@U-y;n=len(x)
    loss=(np.sum(r*r)+RIDGE*np.sum(U*U))/(2*n)
    if not gradient:return loss
    dz=np.zeros_like(z);dp=r@U.T/n
    for k,(i,j) in enumerate(PAIRS):
        dz[:,i]+=dp[:,k]*z[:,j];dz[:,j]+=dp[:,k]*z[:,i]
    gr=np.r_[(dz*b).T@x,(dz*a).T@x]
    gr-=np.sum(gr*rows,axis=1)[:,None]*rows
    gu=(f.T@r+RIDGE*U)/n
    return loss,gr,gu

def optimum(rows,x,y):
    f=features(rows,x)[0]
    return np.linalg.lstsq(np.r_[f,np.sqrt(RIDGE)*np.eye(10)],np.r_[y,np.zeros((10,y.shape[1]))],rcond=None)[0]

def retract(rows,direction,dt):
    new=rows-dt*direction
    return new/np.linalg.norm(new,axis=1)[:,None]

def readout_flow(rows,U,x,y,tau):
    f=features(rows,x)[0];H=(f.T@f+RIDGE*np.eye(10))/len(x);rhs=f.T@y/len(x)
    eigen,Q=np.linalg.eigh(H);assert eigen.min()>0
    star=Q@((Q.T@rhs)/eigen[:,None])
    return star+Q@(np.exp(-tau*eigen)[:,None]*(Q.T@(U-star)))

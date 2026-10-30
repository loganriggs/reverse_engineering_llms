"""Independent metric plumbing; shares the previously checked Wick kernel only."""
import itertools
import numpy as np
from scipy.optimize import linear_sum_assignment
from direct_bilinear_readout import quadratic_coefficients
from quartic_cross_gram import cross_gram

def phi(A,B,x):
    z=(x@A.T)*(x@B.T)
    return np.stack([z[:,i]*z[:,j] for i,j in itertools.combinations_with_replacement(range(len(A)),2)],axis=1)

def factors(A,B):
    return np.array([[A[i],B[i],A[j],B[j]] for i,j in itertools.combinations_with_replacement(range(len(A)),2)])

def matching(A,B,TA,TB):
    d=A.shape[1];diagonal=[k for k,(i,j) in enumerate(itertools.combinations_with_replacement(range(d),2)) if i==j]
    a,b=quadratic_coefficients(A,B),quadratic_coefficients(TA,TB)
    def inner(q,r):return q.T@r+q[diagonal].T@r[diagonal]+np.outer(q[diagonal].sum(0),r[diagonal].sum(0))
    numerator=abs(inner(a,b));denom=np.sqrt(np.outer(np.diag(inner(a,a)),np.diag(inner(b,b))))
    cos=np.divide(numerator,denom,out=np.zeros_like(numerator),where=denom>0);i,j=linear_sum_assignment(-cos)
    return cos[i,j]

def svd_refit(X,y,coef):
    u,s,vh=np.linalg.svd(X,full_matrices=False);keep=s>s[0]*1e-12
    ref=vh[keep].T@((u[:,keep].T@y)/s[keep,None])
    np.testing.assert_allclose(coef,ref,rtol=1e-6,atol=1e-8)
    return int(keep.sum())

def audit_case(A,B,W,TA,TB,TW,x,y,xa,ya,saved,record,teacher_cache=None):
    t=record['tasks'];p=len(A)*(len(A)+1)//2
    assert len(record['population_mse'])==t and len(record['component_cosines'])==len(A)
    if teacher_cache is None:
        TF=factors(TA,TB);TG=cross_gram(TF,TF);energy=np.einsum('tp,pq,tq->t',TW,TG,TW,optimize=True)
    else:TF,TG,energy=teacher_cache
    F=factors(A,B);G=cross_gram(F,F);C=cross_gram(F,TF)
    source=np.einsum('tp,pq,tq->t',W[:t],G,W[:t],optimize=True)-2*np.sum((W[:t]@C)*TW[:t],axis=1)+energy[:t]
    np.testing.assert_allclose(source,record['population_mse'],rtol=1e-7,atol=1e-8)
    total=0.
    for start in range(0,len(x),127):
        residual=phi(A,B,x[start:start+127])@W[:t].T-y[start:start+127,:t];total+=np.sum(residual**2)
    train=float(total/(len(x)*t));np.testing.assert_allclose(train,record['train_mse'],rtol=1e-8,atol=1e-10)
    np.testing.assert_allclose(matching(A,B,TA,TB),record['component_cosines'],rtol=1e-8,atol=1e-10)
    X=phi(A,B,xa);ranks={}
    for k in [2048,8192]:
        assert len(record['transfer'][f'learned_{k}']['errors'])==TW.shape[0]-p
        coef=saved[f'learned_{k}'];rank=svd_refit(X[:k],ya[:k],coef);assert rank==record['transfer'][f'learned_{k}']['rank']
        err=np.einsum('pt,pq,qt->t',coef,G,coef)-2*np.sum((coef.T@C)*TW[p:],axis=1)+energy[p:]
        np.testing.assert_allclose(err,record['transfer'][f'learned_{k}']['errors'],rtol=1e-7,atol=1e-8);ranks[str(k)]=rank
    return dict(train_mse=train,source_max_error=float(np.max(np.abs(source-np.array(record['population_mse'])))),transfer_ranks=ranks)

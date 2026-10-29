"""Algebraic checks for polynomial multiplication kernels and top-width inertia."""
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

from regression_baselines import feature_maps


def factor_symmetric(Q, tol=1e-10):
    eigen,vectors=np.linalg.eigh(Q)
    pos=[np.sqrt(a)*vectors[:,i] for i,a in enumerate(eigen) if a>tol]
    neg=[np.sqrt(-a)*vectors[:,i] for i,a in enumerate(eigen) if a<-tol]
    terms=[]
    for i in range(max(len(pos),len(neg))):
        u=pos[i] if i<len(pos) else np.zeros(len(Q))
        v=neg[i] if i<len(neg) else np.zeros(len(Q))
        terms.append((u+v,u-v))
    return terms, (len(pos),len(neg))


def main():
    root=Path(__file__).parent;out=root/'results/multiplication_geometry'
    out.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(82147);errors=[];full=[]
    # A complete quadratic monomial dictionary spans all degree-four monomials.
    for d in range(2,7):
        quad=list(itertools.combinations_with_replacement(range(d),2))
        quart=list(itertools.combinations_with_replacement(range(d),4));index={k:i for i,k in enumerate(quart)}
        pairs=list(itertools.combinations_with_replacement(range(len(quad)),2))
        M=np.zeros((len(quart),len(pairs)))
        for column,(i,j) in enumerate(pairs):M[index[tuple(sorted(quad[i]+quad[j]))],column]=1
        rank=int(np.linalg.matrix_rank(M));assert rank==math.comb(d+3,4)
        full.append(dict(d=d,quadratic_features=len(quad),symmetric_lift_coordinates=len(pairs),
                         quartic_dimension=len(quart),rank=rank,kernel_dimension=len(pairs)-rank))
    # Verify the constructive upper bound for arbitrary symmetric forms.
    for h in range(1,9):
        for seed in range(10):
            a=rng.normal(size=(h,h));Q=(a+a.T)/2
            factors,inertia=factor_symmetric(Q)
            reconstructed=sum((np.outer(c,e)+np.outer(e,c))/2 for c,e in factors)
            err=float(np.linalg.norm(Q-reconstructed)/np.linalg.norm(Q));assert err<1e-12;errors.append(err)
            assert len(factors)==max(inertia)
    # Same polynomial, different lifts: x^4+y^4, with h=(x^2,xy,y^2).
    Q0=np.diag([1.,0.,1.]);Q1=np.array([[1.,0.,1.],[0.,-2.,0.],[1.,0.,1.]])
    pts=rng.normal(size=(1000,2));z=np.column_stack([pts[:,0]**2,pts[:,0]*pts[:,1],pts[:,1]**2])
    a=np.einsum('ni,ij,nj->n',z,Q0,z);b=np.einsum('ni,ij,nj->n',z,Q1,z)
    np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-12)
    f0,i0=factor_symmetric(Q0);f1,i1=factor_symmetric(Q1)
    assert len(f0)==2 and len(f1)==1
    c=np.array([1.,np.sqrt(2.),1.]);e=np.array([1.,-np.sqrt(2.),1.])
    np.testing.assert_allclose((z@c)*(z@e),a,rtol=1e-12,atol=1e-12)
    # Functional dimensions of the actual frozen teacher hidden-product features.
    teacher_ranks=[]
    for run in ['e1_sweep','e1_d8']:
        protocol=json.loads((root/'results'/run/'protocol.json').read_text())['arguments']
        d=protocol['d'];n=protocol['ntrain'].split(',')[0]
        keys=sorted({tuple(sum(i==j for i in inds) for j in range(d)) for inds in itertools.combinations_with_replacement(range(d),4)})
        for kind in ['tied','untied']:
            for seed in range(5):
                with np.load(root/'results'/run/f'{kind}_{seed}_{n}_tied.npz') as data:
                    w={k.removeprefix('teacher_'):v.copy() for k,v in data.items() if k.startswith('teacher_')}
                K=feature_maps(w,d,keys)['oracle_hidden'];s=np.linalg.svd(K,compute_uv=False)
                teacher_ranks.append(dict(run=run,teacher=kind,seed=seed,columns=K.shape[1],
                                         rank=int(np.sum(s>s[0]*1e-10)),relative_singular_values=(s/s[0]).tolist()))
    result=dict(status='passed',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                full_quadratic_dictionary=full,symmetric_factorization_checks=len(errors),max_relative_error=max(errors),
                same_polynomial_example=dict(initial_inertia=i0,alternative_inertia=i1,initial_fixed_lift_minimum_q=len(f0),alternative_minimum_q=len(f1),
                                             polynomial='x^4+y^4',hidden=['x^2','xy','y^2']),
                teacher_hidden_product_ranks=teacher_ranks)
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='teacher_hidden_product_ranks'}))


if __name__=='__main__':main()

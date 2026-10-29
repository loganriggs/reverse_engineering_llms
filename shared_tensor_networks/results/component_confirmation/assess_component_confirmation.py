"""Gauge-aware quadratic component recovery and counterexample checks."""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from experiment import gram, multiply


def quadratic_map(A,B,keys):
    d=A.shape[-1];unit=[tuple(int(i==j) for i in range(d)) for j in range(d)]
    polys=[multiply(dict(zip(unit,a)),dict(zip(unit,b))) for a,b in zip(A,B)]
    return np.array([[p.get(k,0.) for p in polys] for k in keys])


def main():
    root=Path(__file__).parent;src=root/'results/multitask_confirmation';out=root/'results/component_confirmation'
    out.mkdir(parents=True,exist_ok=False)
    protocol=dict(scope='all 10 trained h4 shared students on shared teachers: 5 teachers x 2 initializations x N256; h10 excluded from exact recovery claim',
                  metric='absolute Gaussian L2 cosine of homogeneous quadratic functions, Hungarian assignment; threshold .95 descriptive only',
                  contrasts='pretraining random initialization and teacher-permutation/scaling positive control',
                  inference='exact identifiability theorem under independent teacher factors and complete task-product span; approximate fits do not automatically imply recovery',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    from multitask import MultiTask
    keys=sorted({tuple(sum(i==j for i in inds) for j in range(8)) for inds in itertools.combinations_with_replacement(range(8),2)})
    H=gram(keys);L=np.linalg.cholesky(H);rows=[];assumptions=[]
    for ts in range(5):
        ev=np.load(src/f'shared_{ts}.npz');A=ev['teacher_A'][0];B=ev['teacher_B'][0]
        K=quadratic_map(A,B,keys);white=L.T@K;norm=np.linalg.norm(white,axis=0);teacher=white/norm
        factor=np.concatenate([A,B],axis=0);singular=np.linalg.svd(factor,compute_uv=False)
        assert singular[-1]>1e-10
        assert np.linalg.matrix_rank(ev['teacher_W'][:12])==10
        assumptions.append(dict(teacher_seed=ts,factor_singular_values=singular.tolist(),factor_condition_number=float(singular[0]/singular[-1]),task_head_rank=10))
        # Verify the rank lemma numerically for every subset of the teacher quadratics.
        for mask in range(1,16):
            coefficients=np.array([(-1.)**j*(j+1) if mask&(1<<j) else 0. for j in range(4)])
            Q=sum(c*(np.outer(a,b)+np.outer(b,a))/2 for c,a,b in zip(coefficients,A,B))
            assert np.linalg.matrix_rank(Q,tol=1e-10)==2*int(np.sum(coefficients!=0))
        for n in [256]:
            for opt in range(2):
                prefix=f'{n}|{opt}|shared'
                initial=MultiTask(8,4,12,True,92000+opt)
                candidates={'trained':(ev[prefix+'|A'],ev[prefix+'|B']),
                            'initial':(initial.A.detach().numpy(),initial.B.detach().numpy()),
                            'gauge_control':(A[[2,0,3,1]]*np.array([2.,-.5,3.,-2.])[:,None],B[[2,0,3,1]])}
                for phase,(a,b) in candidates.items():
                    G=L.T@quadratic_map(a,b,keys);G/=np.linalg.norm(G,axis=0)[None,:]
                    cosine=np.abs(teacher.T@G);i,j=linear_sum_assignment(-cosine);matched=cosine[i,j]
                    if phase=='gauge_control':np.testing.assert_allclose(matched,np.ones(4),rtol=1e-12,atol=1e-12)
                    qt=np.linalg.qr(teacher)[0];qg=np.linalg.qr(G)[0]
                    rows.append(dict(teacher_seed=ts,n=n,optimizer_seed=opt,phase=phase,matched_absolute_cosines=matched.tolist(),
                                     mean_cosine=float(np.mean(matched)),minimum_cosine=float(np.min(matched)),above_095=int(np.sum(matched>.95)),
                                     quadratic_subspace_coverage=float(np.linalg.norm(qt.T@qg)**2/4)))
    # Full task coverage is necessary: h1^2 alone never constrains the other feature.
    rng=np.random.default_rng(88000);x=rng.normal(size=(1000,4))
    h1=x[:,0]*x[:,1];h2=x[:,2]*x[:,3];alternate=x[:,0]*x[:,2]
    teacher_products=np.column_stack([h1*h1,h1*h2,h2*h2])
    alternate_products=np.column_stack([h1*h1,h1*alternate,alternate*alternate])
    np.testing.assert_array_equal(teacher_products[:,0],alternate_products[:,0])
    assert np.linalg.norm(teacher_products[:,1:]-alternate_products[:,1:])>1
    assert np.linalg.norm(h2-alternate)>1
    # Independent factors are necessary: a shared factor permits nontrivial mixing.
    h=np.column_stack([x[:,0]*x[:,1],x[:,0]*x[:,2]])
    g=np.column_stack([h[:,0]+h[:,1],h[:,0]-h[:,1]])
    source=np.column_stack([h[:,0]**2,h[:,0]*h[:,1],h[:,1]**2])
    dest=np.column_stack([g[:,0]**2,g[:,0]*g[:,1],g[:,1]**2])
    change=np.array([[.25,.25,.25],[.5,0.,-.5],[.25,-.25,.25]])
    np.testing.assert_allclose(dest@change,source,rtol=1e-12,atol=1e-12)
    result=dict(status='passed',assumptions=assumptions,rows=rows,rank_lemma_checks=75,
                counterexamples=['incomplete task span leaves unused features unconstrained','dependent factors allow non-permutation mixing'],
                dependent_factor_counterexample_max_error=float(np.max(np.abs(dest@change-source))))
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Factor conditions',[a['factor_condition_number'] for a in assumptions])
    for n in [256]:
        for phase in ['initial','trained']:
            r=[a for a in rows if a['n']==n and a['phase']==phase]
            print(n,phase,'mean cosine',np.mean([a['mean_cosine'] for a in r]),'components above .95',sum(a['above_095'] for a in r),'/40')


if __name__=='__main__':main()

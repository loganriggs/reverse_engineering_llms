"""Independent quadratic-form metrics and exact counterexample coefficients."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from multitask import MultiTask
from experiment import multiply

def embedding(A,B):
    Q=np.array([(np.outer(a,b)+np.outer(b,a))/2 for a,b in zip(A,B)])
    return np.vstack([np.sqrt(2)*Q.reshape(len(Q),-1).T,np.trace(Q,axis1=1,axis2=2)[None,:]])

def main():
    root=Path(__file__).parent;out=root/'results/component_confirmation';src=root/'results/multitask_confirmation'
    result=json.loads((out/'results.json').read_text());maxerr=0.;checks=0
    for row in result['rows']:
        ts,n,opt=row['teacher_seed'],row['n'],row['optimizer_seed'];ev=np.load(src/f'shared_{ts}.npz')
        A=ev['teacher_A'][0];B=ev['teacher_B'][0];teacher=embedding(A,B);teacher/=np.linalg.norm(teacher,axis=0)
        if row['phase']=='trained':a,b=ev[f'{n}|{opt}|shared|A'],ev[f'{n}|{opt}|shared|B']
        elif row['phase']=='initial':
            initial=MultiTask(8,4,12,True,92000+opt);a,b=initial.A.detach().numpy(),initial.B.detach().numpy()
        else:a,b=A[[2,0,3,1]]*np.array([2.,-.5,3.,-2.])[:,None],B[[2,0,3,1]]
        learned=embedding(a,b);learned/=np.linalg.norm(learned,axis=0)
        assert np.linalg.matrix_rank(teacher,tol=1e-10)==4 and np.linalg.matrix_rank(learned,tol=1e-10)==4
        cos=np.abs(teacher.T@learned);i,j=linear_sum_assignment(-cos);matched=cos[i,j]
        err=float(np.max(np.abs(matched-row['matched_absolute_cosines'])));assert err<1e-12;maxerr=max(maxerr,err)
        q1=np.linalg.qr(teacher)[0];q2=np.linalg.qr(learned)[0]
        coverage=float(np.linalg.norm(q1.T@q2)**2/4);assert abs(coverage-row['quadratic_subspace_coverage'])<1e-12;checks+=1
    h1={(1,1,0,0):1};h2={(0,0,1,1):1};g2={(1,0,1,0):1}
    teacher_products=[multiply(h1,h1),multiply(h1,h2),multiply(h2,h2)]
    alternate_products=[multiply(h1,h1),multiply(h1,g2),multiply(g2,g2)]
    assert teacher_products[0]==alternate_products[0]
    assert teacher_products[1]!=alternate_products[1] and teacher_products[2]!=alternate_products[2]
    h1={(1,1,0):1};h2={(1,0,1):1};g1={**h1,**h2};g2={**h1,(1,0,1):-1}
    source=[multiply(h1,h1),multiply(h1,h2),multiply(h2,h2)]
    dest=[multiply(g1,g1),multiply(g1,g2),multiply(g2,g2)]
    change=np.array([[1,1,1],[2,0,-2],[1,-1,1]])
    for j in range(3):
        for key in set().union(*source,*dest):
            assert sum(int(change[i,j])*dest[i].get(key,0) for i in range(3))==4*source[j].get(key,0)
    audit=dict(status='passed',independent_quadratic_geometry_checks=checks,max_cosine_error=maxerr,exact_polynomial_counterexamples=2,
               source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),results_sha256=hashlib.sha256((out/'results.json').read_bytes()).hexdigest())
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit))

if __name__=='__main__':main()

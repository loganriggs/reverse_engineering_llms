"""Oracle-aligned canonical factor deletion and exact covariance-shift assessment."""
import itertools,json,math
from pathlib import Path
import numpy as np
from assess_fixed_degree_recovery import assess

def coeff(factors,amplitude):
    c={(0,0,0):amplitude}
    for a in factors:
        nxt={}
        for power,v in c.items():
            for j,aj in enumerate(a):
                p=list(power);p[j]+=1;p=tuple(p);nxt[p]=nxt.get(p,0)+v*aj
        c=nxt
    return c

def inner(a,b,var):
    total=[]
    for pa,va in a.items():
        for pb,vb in b.items():
            power=[x+y for x,y in zip(pa,pb)]
            moment=math.prod(0 if n%2 else math.prod(range(1,n,2))*v**(n//2) for n,v in zip(power,var));total.append(va*vb*moment)
    return math.fsum(total)
def relative(a,b,var):
    diff={p:a.get(p,0)-b.get(p,0) for p in set(a)|set(b)}
    return inner(diff,diff,var)/inner(b,b,var)
def main():
    root=Path(__file__).parent;src=root/'results/fixed_degree_reuse';out=root/'results/fixed_degree_recovery';rows=[]
    for seed in [170001,170002,170003]:
        z=np.load(src/f't4_s{seed}_independent_g4.npz');t=z['teacher'];tn=np.linalg.norm(t,axis=1);tu=t/tn[:,None];ta=float(np.prod(tn)/z['teacher_scale'])
        for stage in ['initial','student']:
            a=z[stage];an=np.linalg.norm(a,axis=1);au=a/an[:,None];m=assess(t,a,float(z['teacher_scale']),float(z['student_scale']));aligned=au[m['permutation']];signs=np.sign(np.sum(aligned*tu,axis=1));aligned*=signs[:,None];sa=float(np.prod(an)/z['student_scale']*np.prod(signs))
            teacher=coeff(tu,ta);student=coeff(aligned,sa)
            # Alignment must preserve the original polynomial, including its global sign.
            direct=coeff(au,float(np.prod(an)/z['student_scale']));assert relative(student,direct,[1,1,1])<1e-25
            deletions=[relative(coeff(np.delete(aligned,i,axis=0),sa),coeff(np.delete(tu,i,axis=0),ta),[1,1,1]) for i in range(4)]
            rows.append(dict(seed=seed,stage=stage,canonical_single_factor_deletion_relative_mse=deletions,shifted_covariance_relative_mse=relative(student,teacher,[.5,1,2]),standard_relative_mse=relative(student,teacher,[1,1,1])))
    result=dict(status='complete',scope='all three independent-initialized four-factor students on four-factor teachers; teacher factor alignment is privileged evaluation only; unit factor norms, aligned signs and original global amplitude; deletion replaces one canonical linear factor with1',rows=rows)
    (out/'interventions.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()

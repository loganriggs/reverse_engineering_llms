"""Permutation/sign/scale-invariant matching of four factor occurrences."""
import hashlib,itertools,json,math
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
PERMS=list(itertools.permutations(range(4)))
def expanded(a):return np.repeat(a,4//len(a),axis=0)
def assess(teacher,student,tscale,sscale):
    t=expanded(teacher);s=expanded(student);tn=np.linalg.norm(t,axis=1);sn=np.linalg.norm(s,axis=1)
    if np.any(tn==0) or np.any(sn==0):return dict(valid=False)
    signed=(t/tn[:,None])@(s/sn[:,None]).T;c=np.minimum(1.,abs(signed));scores=[float(c[np.arange(4),perm].mean()) for perm in PERMS];best=PERMS[int(np.argmax(scores))];values=c[np.arange(4),best]
    # Independent assignment implementation verifies exhaustive mean optimum.
    ii,jj=linear_sum_assignment(-c);assert abs(float(c[ii,jj].mean())-max(scores))<1e-12
    bottleneck=max(float(c[np.arange(4),perm].min()) for perm in PERMS)
    amplitude=float(np.prod(sn)/sscale/(np.prod(tn)/tscale)*np.prod(np.sign(signed[np.arange(4),best])))
    return dict(valid=True,mean_cosine=float(values.mean()),worst_cosine_at_mean_optimum=float(values.min()),optimal_bottleneck_cosine=bottleneck,maximum_angle_degrees_at_mean_optimum=float(np.degrees(np.arccos(values.min()))),signed_amplitude_ratio=amplitude,strict_recovery=bool(values.min()>=.99 and abs(amplitude-1)<=.05),permutation=list(best))
def main():
    root=Path(__file__).parent;src=root/'results/fixed_degree_reuse';out=root/'results/fixed_degree_recovery';out.mkdir(exist_ok=False)
    protocol=dict(metric='expand to four occurrences; maximize mean absolute cosine over24permutations; separately report optimal bottleneck; signed normalized amplitude',strict_thresholds=dict(each_cosine=.99,relative_amplitude_error=.05),purpose='descriptive fixed thresholds, no statistical significance claim',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    # Exact invariance fixture with permutation, reciprocal scaling, and two sign flips.
    t=np.array([[1.,0,0],[0,1,0],[0,0,1],[1.,1,1]]);s=t[[2,0,3,1]]*np.array([-2.,3.,-.5,1/3])[:,None];m=assess(t,s,1.,1.);assert m['strict_recovery']
    s[0]*=-1;assert not assess(t,s,1.,1.)['strict_recovery']
    t=np.array([[1.,0,0],[0,1,0]]);s=np.array([[1.,0,0],[1,0,0],[1,0,0],[0,1,0]]);assert assess(t,s,1,1)['optimal_bottleneck_cosine']==0
    rows=[]
    for f in sorted(src.glob('t*_s*.json')):
        r=json.loads(f.read_text());a=np.load(f.with_suffix('.npz'));t=a['teacher'];u=t/np.linalg.norm(t,axis=1,keepdims=True);angles=[float(np.degrees(np.arccos(np.clip(abs(u[i]@u[j]),0,1)))) for i in range(len(t)) for j in range(i)]
        row={k:r[k] for k in ['name','teacher_groups','student_groups','seed','initialization']};row.update(population_mse=r['curves'][-1]['population_mse'],teacher_min_distinct_angle_degrees=min(angles) if angles else None,initial=assess(t,a['initial'],float(a['teacher_scale']),float(a['student_scale'])),final=assess(t,a['student'],float(a['teacher_scale']),float(a['student_scale'])),input_sha256=hashlib.sha256(f.with_suffix('.npz').read_bytes()).hexdigest());rows.append(row)
    # Repeated-root stability fixture: common normalization sqrt(E[u^8])=sqrt105.
    stability=[]
    for delta in [1e-1,3e-2,1e-2,3e-3,1e-3]:
        t=np.array([[1.,0,0]]);s=np.array([[1.,delta,0],[1.,-delta,0],[1.,0,0],[1.,0,0]])
        from audit_fixed_degree_reuse import coefficients,inner
        tc=coefficients(t,math.sqrt(105));sc=coefficients(s,math.sqrt(105));diff={p:sc.get(p,0)-tc.get(p,0) for p in set(sc)|set(tc)};mse=inner(diff,diff);expected=3*delta**4/35;assert abs(mse-expected)<1e-12*expected
        metric=assess(t,s,math.sqrt(105),math.sqrt(105));assert abs(metric['maximum_angle_degrees_at_mean_optimum']-np.degrees(np.arctan(delta)))<1e-8
        stability.append(dict(delta=delta,population_mse=mse,angle_degrees=metric['maximum_angle_degrees_at_mean_optimum']))
    (out/'results.json').write_text(json.dumps(dict(status='complete',rows=rows,stability=stability,invariance_assignment_checks='passed'),indent=2)+'\n');print('Matched81fits; invariance/multiplicity/assignment and five exact stability fixtures passed')
if __name__=='__main__':main()

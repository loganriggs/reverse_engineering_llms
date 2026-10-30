"""Post-hoc quadratic feature-span diagnostic; no retraining or seed selection."""
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
root=Path('results/task_coverage_h64');partial=json.loads(Path('results/h64_partial_recovery/analysis.json').read_text());rows=json.loads((root/'evaluation/results.json').read_text())
assert partial['input_sha256'][str(root/'evaluation/results.json')]==hashlib.sha256((root/'evaluation/results.json').read_bytes()).hexdigest()
def matrix(A,B):return (A[:,:,None]*B[:,None,:]+B[:,:,None]*A[:,None,:])/2
def inner(Q,R):return 2*np.einsum('iab,jab->ij',Q,R)+np.outer(np.trace(Q,axis1=1,axis2=2),np.trace(R,axis1=1,axis2=2))
def span(Q,T):
 G=inner(Q,Q);C=inner(Q,T);H=inner(T,T);gs=np.sqrt(np.diag(G));ts=np.sqrt(np.diag(H));G/=gs[:,None]*gs[None,:];C/=gs[:,None]*ts[None,:];H/=ts[:,None]*ts[None,:]
 lam,U=np.linalg.eigh(G);keep=lam>max(lam)*1e-12;coef=U[:,keep]@((U[:,keep].T@C)/lam[keep,None]);error=1-np.sum(C*coef,axis=0);assert min(error)>-1e-8
 i,j=linear_sum_assignment(-abs(C));cos=abs(C[i,j]);missing=j[cos<=.99]
 return dict(rank=int(sum(keep)),normalized_teacher_residuals=np.maximum(error,0).tolist(),mean_residual=float(np.mean(np.maximum(error,0))),missing_count=len(missing),missing_mean_residual=float(np.mean(np.maximum(error[missing],0))) if len(missing) else None,max_residual=float(max(error)),matched_cosines=cos.tolist())
# Coordinate change preserves the span, while removal of a direction changes it.
rng=np.random.default_rng(268001);raw=rng.normal(size=(4,8,8));Q=(raw+raw.transpose(0,2,1))/2;mix=np.eye(4)+.1*rng.normal(size=(4,4));check=span(np.einsum('ij,jab->iab',mix,Q),Q);assert check['max_residual']<1e-10
check_drop=span(Q[:3],Q);assert check_drop['normalized_teacher_residuals'][3]>.01
records=[]
for r in rows:
 td=root/str(r['teacher']);checkpoint=td/f'n{r["n"]}_t{r["tasks"]}'/f'{r["initialization"]}.npz';assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==r['checkpoint_sha256']
 with np.load(td/'teacher_factors.npz') as z:T=matrix(z['A'],z['B'])
 with np.load(checkpoint) as z:Q=matrix(z['A'].astype(float),z['B'].astype(float))
 result=span(Q,T);np.testing.assert_allclose(result['matched_cosines'],r['component_cosines'],rtol=1e-10,atol=1e-10)
 records.append(dict(name=r['name'],n=r['n'],tasks=r['tasks'],teacher=r['teacher'],initialization=r['initialization'],**result))
summary=[]
for n in [8192,32768]:
 for tasks in [520,2080]:
  runs=[r for r in records if r['n']==n and r['tasks']==tasks and r['initialization']!='teacher'];teacher_means=[]
  for t in sorted({r['teacher'] for r in runs}):teacher_means.append(float(np.mean([r['mean_residual'] for r in runs if r['teacher']==t])))
  missing=[r['missing_mean_residual'] for r in runs if r['missing_count']]
  summary.append(dict(n=n,tasks=tasks,median_teacher_mean_span_residual=float(np.median(teacher_means)),range_missing_direction_mean_residual=[min(missing),max(missing)] if missing else None,ranks=sorted({r['rank'] for r in runs})))
out=Path('results/h64_feature_span');out.mkdir(exist_ok=False)
result=dict(scope='Exploratory post-hoc population L2 projection of teacher quadratics onto learned quadratic span; no new learning runs',cutoff=1e-12,checks=dict(invertible_mix_max_error=check['max_residual'],removed_direction_residual=check_drop['normalized_teacher_residuals'][3]),records=records,summary=summary,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),audited_results_sha256=hashlib.sha256((root/'evaluation/results.json').read_bytes()).hexdigest())
(out/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
for s in summary:print(json.dumps(s),flush=True)

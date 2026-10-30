"""Real-shape smoke evaluation resource check; no scientific trained-run claims."""
import hashlib,json,resource,time
from pathlib import Path
import numpy as np
from regenerated_coverage import load_readout_chunks
from coverage_evaluation import teacher_geometry,candidate_geometry,adaptation,features,components,population_errors
from coverage_audit import phi,svd_refit,matching
root=Path('/tmp/task_coverage_h64_smoke/256101');meta=json.loads((root/'teacher.json').read_text());TW=load_readout_chunks(root,meta['readout_chunks'])
with np.load(root/'teacher_factors.npz') as f:TA,TB=f['A'],f['B']
with np.load(root/'n8192_t520/257001.npz') as f:A,B,W=[f[k].astype(float) for k in ['A','B','W']]
with np.load(root/'adaptation.npz') as f:x,y=f['x'],f['y']
assert A.shape==(64,128) and TW.shape==(2084,2080) and x.shape==(8192,128)
times={};started=time.perf_counter();tg=teacher_geometry(TA,TB,TW);times['teacher_geometry']=time.perf_counter()-started;np.testing.assert_allclose(tg['energy'],1,atol=1e-9)
print('Teacher geometry',times['teacher_geometry'],flush=True)
started=time.perf_counter();cg=candidate_geometry(A,B,tg);times['candidate_geometry']=time.perf_counter()-started
print('Candidate geometry',times['candidate_geometry'],flush=True)
np.testing.assert_allclose(features(A,B,x[:32]),phi(A,B,x[:32]),atol=1e-12)
np.testing.assert_allclose(components(A,B,TA,TB),matching(A,B,TA,TB),atol=1e-10)
records=[]
for budget in [2048,8192]:
 started=time.perf_counter();coef,record=adaptation(A,B,x,y,cg['G'],cg['C'],TW[2080:],tg['energy'][2080:],budget);times[f'fit_{budget}']=time.perf_counter()-started
 print('Fit',budget,times[f'fit_{budget}'],flush=True)
 started=time.perf_counter();rank=svd_refit(phi(A,B,x[:budget]),y[:budget],coef);times[f'independent_svd_{budget}']=time.perf_counter()-started;assert rank==record['rank']
 errors=np.einsum('pt,pq,qt->t',coef,cg['G'],coef)-2*np.sum((coef.T@cg['C'])*TW[2080:],axis=1)+tg['energy'][2080:]
 np.testing.assert_allclose(errors,record['errors'],rtol=1e-7,atol=1e-8);assert np.isfinite(errors).all() and min(errors)>-1e-7
 records.append(dict(budget=budget,rank=rank,population_error_replay=True,coefficient_svd_agreement=True))
source=population_errors(W[:520].T,cg['G'],cg['C'],TW[:520],tg['energy'][:520]);assert np.isfinite(source).all() and min(source)>-1e-7
out=Path('results/h64_analysis_preflight');out.mkdir(exist_ok=True)
result=dict(scope='Two-step smoke checkpoint only; not scientific h64 outcomes',times_seconds=times,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
 records=records,source_tasks=520,input_dimension=128,hidden_width=64,features=2080,source_sha256={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in ['benchmark_h64_analysis.py','coverage_evaluation.py','coverage_audit.py','quartic_cross_gram.py']},
 input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [root/'teacher.json',root/'teacher_factors.npz',root/'adaptation.npz',root/'n8192_t520/257001.npz']})
(out/'checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)

import hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from depth3_causal_model import coefficients,normalize,teacher,feature_metrics,G,L
root=Path('results/depth3_causal_training');root.mkdir(exist_ok=False)
(root/'protocol.json').write_text(json.dumps({n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in ['DEPTH3_CAUSAL_TRAINING_PLAN.md','depth3_causal_model.py','run_depth3_causal_training.py']},indent=2)+'\n')
records=[]
for seed in range(263101,263106):
 tv=teacher(seed);targets=[coefficients(tv,False)[0],coefficients(tv,True)[0]]
 for initial in [264001,264002]:
  rng=np.random.default_rng(initial);v0=normalize(rng.normal(size=17)/np.sqrt(2))
  for causal in [False,True]:
   cache={}
   def evaluate(v):
    if 'v' not in cache or not np.array_equal(v,cache['v']):
     rr=[];jj=[]
     for mode in range(2 if causal else 1):
      c,J=coefficients(v,bool(mode));rr.append(((c-targets[mode])@L).ravel());jj.append(np.einsum('tik,ij->tjk',J,L).reshape(-1,17))
     cache.update(v=v.copy(),r=np.concatenate(rr),J=np.concatenate(jj))
    return cache
   started=time.perf_counter();fit=least_squares(lambda v:evaluate(v)['r'],v0,jac=lambda v:evaluate(v)['J'],method='trf',max_nfev=300,ftol=1e-10,xtol=1e-10,gtol=1e-10)
   errors=[]
   for mode in [False,True]:
    error=coefficients(fit.x,mode)[0]-targets[int(mode)];errors.append(np.einsum('ti,ij,tj->t',error,G,error).tolist())
   record=dict(seed=seed,initial=initial,causal=causal,teacher=tv.tolist(),start=v0.tolist(),final=fit.x.tolist(),status=int(fit.status),success=bool(fit.success),nfev=fit.nfev,njev=fit.njev,seconds=time.perf_counter()-started,ordinary_mse=errors[0],ablated_mse=errors[1],loss=float(fit.cost),gradient_norm=float(np.linalg.norm(fit.grad)),**feature_metrics(fit.x,tv))
   records.append(record);(root/'results.json').write_text(json.dumps(records,indent=2)+'\n');print(seed,initial,causal,fit.nfev,float(np.mean(errors[0])),record['matched_cosines'],flush=True)
(root/'complete.json').write_text(json.dumps(dict(status='trained_not_audited',fits=len(records)))+'\n')

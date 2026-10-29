"""Exact random-subset expectation, conditional on measured channel contributions."""
import itertools,json,hashlib
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;V=P/'validation'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def formulas(n,k,rho):
 p=k/n;q=k*(k-1)/(n*(n-1));s=q+(p-q)*rho
 return {'expected_raw_relative_squared':1-2*p+s,'ensemble_optimal_scalar':p/s,'expected_error_at_ensemble_scalar':1-p*p/s}
def run():
 d=json.loads((V/'product_subset_results.json').read_text());a=json.loads((V/'product_subset_audit.json').read_text());assert a['status']=='passed' and a['result_sha256']==sha(V/'product_subset_results.json')
 # Exhaustively verify the expectation formulas on signed vector contributions.
 rng=np.random.default_rng(9003);checks=[]
 for n in [3,4,6]:
  C=rng.normal(size=(n,7));F=C.sum(0);E=F@F;rho=float(np.sum(C*C)/E)
  for k in range(1,n+1):
   pred=formulas(n,k,rho);sums=np.array([C[list(ix)].sum(0) for ix in itertools.combinations(range(n),k)])
   raw=float(np.mean(np.sum((sums-F)**2,axis=1))/E);comp=float(np.mean(np.sum((pred['ensemble_optimal_scalar']*sums-F)**2,axis=1))/E)
   assert abs(raw-pred['expected_raw_relative_squared'])<1e-12 and abs(comp-pred['expected_error_at_ensemble_scalar'])<1e-12
   checks.append({'n':n,'k':k,'max_error':max(abs(raw-pred['expected_raw_relative_squared']),abs(comp-pred['expected_error_at_ensemble_scalar']))})
 rows=[]
 for inv in d['inventories']:
  for k in [128,512,2048,2304,4608]:
   f=formulas(4608,k,inv['diagonal_over_total']);rs=[r for r in d['rows'] if r['space']==inv['space'] and r['size']==k and r['method'].startswith('random')]
   rows.append({'space':inv['space'],'size':k,**f,'observed_random_raw':[r['raw_relative_squared'] for r in rs],'observed_per_subset_fitted_scalar':[r['scalar'] for r in rs],'observed_per_subset_fitted_error':[r['compensated_relative_squared'] for r in rs]})
 out={'status':'passed','source_sha256':sha(__file__),'result_sha256':sha(V/'product_subset_results.json'),'audit_sha256':sha(V/'product_subset_audit.json'),'exhaustive_formula_checks':checks,'rows':rows,'scope':'Posthoc exact expectation derivation, not a preregistered prediction or extra native experiment. Two random subsets are illustrative, not a distribution estimate. Ensemble-optimal common scalar differs from each subset optimum; no equality claim for their errors.'}
 (V/'product_subset_random_theory.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':run()

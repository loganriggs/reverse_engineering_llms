"""Audit tiny complete orchestration and unchanged completed-case resume."""
import hashlib,itertools,json,subprocess,sys
from pathlib import Path
import numpy as np
from numpy.polynomial.hermite import hermgauss
from chunked_checkpoint import load_checkpoint
from coverage_audit import phi,svd_refit
from coverage_evaluation import adaptation,population_errors
from quartic_cross_gram import product_factors,cross_gram
root=Path('/tmp/task_coverage_h128_smoke')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert json.loads((root/'complete.json').read_text())['fits']==12
before={str(p):sha(p) for p in root.rglob('*') if p.is_file() and p.name!='worker.lock'}
r=subprocess.run([sys.executable,'run_task_coverage_h128.py','--root',str(root),'--smoke'],capture_output=True,text=True);assert r.returncode==0,r.stderr
assert before=={str(p):sha(p) for p in root.rglob('*') if p.is_file() and p.name!='worker.lock'}
A,B,W=load_checkpoint(root/'266101'/'teacher');F=product_factors(A,B,np.ones(2));G=cross_gram(F,F)
nodes,weights=hermgauss(5);ix=np.array(list(itertools.product(range(5),repeat=4)));gx=np.sqrt(2)*nodes[ix];gw=np.prod(weights[ix]/np.sqrt(np.pi),axis=1);tf=phi(A,B,gx)
np.testing.assert_allclose(np.sum(gw[:,None]*(tf@W.T)**2,axis=0),1,rtol=1e-12,atol=1e-12)
with np.load(root/'266101'/'adaptation.npz') as z:ax=z['x'];ay=z['y']
records=[]
for n in [128,512]:
 for t in [1,3]:
  for init in ['267001','267002','teacher']:
   case=root/'266101'/f'n{n}_t{t}'/init;a,b,w=load_checkpoint(case/'checkpoint')
   expected=W[:3].astype(np.float32) if init=='teacher' else load_checkpoint(root/'initials'/init)[2]
   np.testing.assert_array_equal(w[t:],expected[t:]);a,b,w=[q.astype(float) for q in [a,b,w]]
   FF=product_factors(a,b,np.ones(2));GG=cross_gram(FF,FF);C=cross_gram(FF,F)
   error=population_errors(w[:t].T,GG,C,W[:t],np.ones(t));delta=phi(a,b,gx)@w[:t].T-tf@W[:t].T
   np.testing.assert_allclose(error,np.sum(gw[:,None]*delta**2,axis=0),rtol=1e-10,atol=1e-10)
   for budget in [16,64]:
    coef,metrics=adaptation(a,b,ax,ay,GG,C,W[3:],np.ones(4),budget);assert svd_refit(phi(a,b,ax[:budget]),ay[:budget],coef)==metrics['rank']
    delta=phi(a,b,gx)@coef-tf@W[3:].T;np.testing.assert_allclose(metrics['errors'],np.sum(gw[:,None]*delta**2,axis=0),rtol=1e-10,atol=1e-10)
   records.append(dict(n=n,tasks=t,initialization=init,source_quadrature=True,transfer_svd_and_quadrature=True,inactive_unchanged=True))
out=Path('results/h128_runner_smoke_audit');out.mkdir(exist_ok=False)
result=dict(scope='Tiny d4/h2 CPU grid; not scientific h128 outcomes',fits=12,completed_resume_byte_unchanged=True,records=records,protocol=json.loads((root/'protocol.json').read_text()),artifact_hashes=before,sources={n:sha(Path(n)) for n in ['audit_h128_runner_smoke.py','run_task_coverage_h128.py','streamed_training_case.py','chunked_checkpoint.py']})
(out/'checks.json').write_text(json.dumps(result,indent=2)+'\n');print('12 cases audited; resume unchanged',flush=True)

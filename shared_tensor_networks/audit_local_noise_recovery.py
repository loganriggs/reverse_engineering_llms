import hashlib,json
from pathlib import Path
import numpy as np
root=Path('results/local_noise_recovery');data=json.loads((root/'results.json').read_text());records=data['records'];assert len(records)==450
assert json.loads((root/'complete.json').read_text())['fits']==450
for name,digest in json.loads((root/'protocol.json').read_text())['source_sha256'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest
expected={(s,n,d,sigma) for s in range(262101,262106) for n in [16,64,256] for d in range(10) for sigma in [.001,.0001,.00001]}
assert {(r['seed'],r['n'],r['draw'],r['sigma']) for r in records}==expected
base=np.eye(4)[[0,2,1,3]];Q=np.eye(16)[:,[4*i+j for i in range(4) for j in range(4) if base[i,j]==0]]
def features(v,x):
 rows=(base.ravel()+Q@v).reshape(4,4);rows/=np.linalg.norm(rows,axis=1)[:,None]
 z1=(x@rows[0])*(x@rows[2]);z2=(x@rows[1])*(x@rows[3])
 return np.column_stack([z1*z1,z1*z2,z2*z2])
def residual(v,x,y):
 phi=features(v,x);u,s,vh=np.linalg.svd(phi,full_matrices=False);coef=(vh.T/s)@(u.T@y)
 return (phi@coef-y).ravel(),coef
checks=[];initial_errors=[]
for seed in range(262101,262106):
 rng=np.random.default_rng(seed);xall=rng.normal(size=(256,4));W=rng.normal(size=(3,3))
 for n in [16,64,256]:
  fixture=np.load(root/f'fixture_{seed}_{n}.npz');noise=rng.normal(size=(10,n,3));x=xall[:n]
  for key,value in [('x',x),('W',W),('noise',noise)]:np.testing.assert_array_equal(fixture[key],value)
  y=features(np.zeros(12),x)@W;step=1e-6
  fd=np.column_stack([(residual(step*np.eye(12)[k],x,y)[0]-residual(-step*np.eye(12)[k],x,y)[0])/(2*step) for k in range(12)])
  A=fixture['A'];err=float(np.linalg.norm(fd-A)/max(1.,np.linalg.norm(A)));assert err<1e-6;initial_errors.append(err)
  # Direct augmented design gives linear response without projecting noise explicitly.
  phi=features(np.zeros(12),x);u,s,vh=np.linalg.svd(A,full_matrices=False);inverse=(vh.T/s)@u.T
  for r in [r for r in records if r['seed']==seed and r['n']==n]:
   v=np.array(r['coordinates']);target=y+r['sigma']*noise[r['draw']];res,coef=residual(v,x,target)
   np.testing.assert_allclose(coef,r['readout'],rtol=1e-8,atol=1e-9)
   np.testing.assert_allclose(res@res/2,r['loss'],rtol=1e-7,atol=1e-16)
   linear=inverse@(r['sigma']*noise[r['draw']].ravel());np.testing.assert_allclose(linear,r['linear_coordinates'],rtol=1e-8,atol=1e-13)
   np.testing.assert_allclose(v@v,r['squared_error'],rtol=1e-12)
   np.testing.assert_allclose(linear@linear,r['linear_squared_error'],rtol=1e-10)
   np.testing.assert_allclose(np.linalg.norm(v-linear)/max(np.linalg.norm(linear),1e-30),r['relative_linearization_error'],rtol=1e-7,atol=1e-12)
   np.testing.assert_allclose(r['sigma']**2*np.sum(1/s**2),r['predicted_risk'],rtol=1e-10)
   final_fd=np.column_stack([(residual(v+step*np.eye(12)[k],x,target)[0]-residual(v-step*np.eye(12)[k],x,target)[0])/(2*step) for k in range(12)])
   gn=float(np.linalg.norm(final_fd.T@res));assert abs(gn-r['gradient_norm'])<1e-7
   checks.append(dict(seed=seed,n=n,draw=r['draw'],sigma=r['sigma'],gradient_norm_fd=gn,gradient_norm_discrepancy=abs(gn-r['gradient_norm'])))
summary=[];paired=0
for seed in range(262101,262106):
 for n in [16,64,256]:
  for draw in range(10):
   rr=sorted([r for r in records if (r['seed'],r['n'],r['draw'])==(seed,n,draw)],key=lambda r:-r['sigma'])
   paired+=int(all(rr[i+1]['relative_linearization_error']<rr[i]['relative_linearization_error'] for i in range(2)))
for n in [16,64,256]:
 for sigma in [.001,.0001,.00001]:
  rr=[r for r in records if r['n']==n and r['sigma']==sigma]
  # Average within each fixture before taking a median across five fixtures.
  means=[np.mean([r['relative_linearization_error'] for r in rr if r['seed']==seed]) for seed in range(262101,262106)]
  ratios=[sum(r['squared_error'] for r in rr if r['seed']==seed)/sum(r['predicted_risk'] for r in rr if r['seed']==seed) for seed in range(262101,262106)]
  summary.append(dict(n=n,sigma=sigma,median_fixture_discrepancy=float(np.median(means)),max_discrepancy=max(r['relative_linearization_error'] for r in rr),median_risk_ratio=float(np.median(ratios))))
result=dict(status='passed',fits=450,successful_solver_statuses=sum(r['success'] for r in records),paired_discrepancy_decreases=paired,paired_sequences=150,
 max_initial_fd_error=max(initial_errors),max_final_gradient_discrepancy=max(c['gradient_norm_discrepancy'] for c in checks),max_gradient_norm=max(r['gradient_norm'] for r in records),
 summary=summary,checks=checks,results_sha256=hashlib.sha256((root/'results.json').read_bytes()).hexdigest())
(root/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['checks']},indent=2))

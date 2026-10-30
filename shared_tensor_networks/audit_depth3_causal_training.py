import hashlib,itertools,json
from pathlib import Path
import numpy as np
from numpy.polynomial.hermite import hermgauss
from depth3_causal_model import coefficients
root=Path('results/depth3_causal_training');records=json.loads((root/'results.json').read_text());assert len(records)==20
assert {(r['seed'],r['initial'],r['causal']) for r in records}=={(s,i,c) for s in range(263101,263106) for i in [264001,264002] for c in [False,True]}
for name,digest in json.loads((root/'protocol.json').read_text()).items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest
nodes,weights=hermgauss(5);nodes*=np.sqrt(2);weights/=np.sqrt(np.pi)
idx=np.array(list(itertools.product(range(5),repeat=4)));x=nodes[idx];weight=np.prod(weights[idx],axis=1);z=np.column_stack([x[:,0]*x[:,1],x[:,2]*x[:,3]])
def hidden(v,z):return (z@v[:4].reshape(2,2).T)*(z@v[4:8].reshape(2,2).T)
def forward(v,z,ablate):
 h=hidden(v,z)
 if ablate:h[:,0]=0
 return np.column_stack([h[:,0]**2,h[:,0]*h[:,1],h[:,1]**2])@v[8:].reshape(3,3).T
probe=np.column_stack([np.ones(5),np.arange(-2,3)]);V=np.array([[a**(4-j)*b**j for j in range(5)] for a,b in probe])
checks=[]
for r in records:
 tv=np.array(r['teacher']);start=np.array(r['start']);v=np.array(r['final'])
 for p in [tv,start]:np.testing.assert_allclose(np.sum(weight[:,None]*forward(p,z,False)**2,axis=0),1,rtol=1e-10)
 # Teacher hidden quadratics are exactly the predeclared mixture, before readout normalization.
 rng=np.random.default_rng(r['seed']);R=np.eye(2)+.2*rng.normal(size=(2,2))
 expected=np.column_stack([z[:,0]**2-z[:,1]**2,2*z[:,0]*z[:,1]])@R.T
 np.testing.assert_allclose(hidden(tv,z),expected,atol=1e-12,rtol=1e-12)
 errors=[];max_jac=0.
 for ablate in [False,True]:
  pred=forward(v,z,ablate);target=forward(tv,z,ablate);mse=np.sum(weight[:,None]*(pred-target)**2,axis=0);errors.append(mse)
  np.testing.assert_allclose(mse,r['ablated_mse' if ablate else 'ordinary_mse'],atol=1e-12,rtol=1e-8)
  for p in [start,v]:
   C,J=coefficients(p,ablate);np.testing.assert_allclose(V@C.T,forward(p,probe,ablate),rtol=1e-10,atol=1e-9)
   direct=np.einsum('ni,tik->ntk',V,J)
   fd=[]
   for k in range(17):
    step=1e-5*max(1.,abs(p[k]));plus=p.copy();minus=p.copy();plus[k]+=step;minus[k]-=step
    fd.append((forward(plus,probe,ablate)-forward(minus,probe,ablate))/(2*step))
   fd=np.stack(fd,axis=2);err=float(np.linalg.norm(fd-direct)/max(1.,np.linalg.norm(direct)));assert err<1e-7;max_jac=max(max_jac,err)
 expected_loss=.5*sum(np.sum(e) for e in errors[:2 if r['causal'] else 1]);np.testing.assert_allclose(expected_loss,r['loss'],rtol=1e-8,atol=1e-12)
 f=hidden(v,z)*np.sqrt(weight[:,None]);t=hidden(tv,z)*np.sqrt(weight[:,None]);cos=np.abs(f.T@t)/np.outer(np.linalg.norm(f,axis=0),np.linalg.norm(t,axis=0))
 np.testing.assert_allclose(np.diag(cos),r['ordered_cosines'],atol=1e-12)
 match=np.diag(cos) if np.trace(cos)>=cos[0,1]+cos[1,0] else np.array([cos[0,1],cos[1,0]])
 np.testing.assert_allclose(match,r['matched_cosines'],atol=1e-12)
 def basis(a):u,s,vh=np.linalg.svd(a,full_matrices=False);return u[:,s>s[0]*1e-10]
 q1=basis(f);q2=basis(t);distance2=q1.shape[1]+q2.shape[1]-2*np.sum((q1.T@q2)**2)
 np.testing.assert_allclose(distance2,r['span_projector_distance']**2,atol=1e-12)
 checks.append(dict(seed=r['seed'],initial=r['initial'],causal=r['causal'],max_relative_jacobian_error=max_jac))
summary=[]
for causal in [False,True]:
 rr=[r for r in records if r['causal']==causal]
 means=lambda key:[np.mean([np.mean(r[key]) for r in rr if r['seed']==seed]) for seed in range(263101,263106)]
 summary.append(dict(causal=causal,source_median=float(np.median(means('ordinary_mse'))),ablated_median=float(np.median(means('ablated_mse'))),
 source_exact=int(sum(np.mean(r['ordinary_mse'])<1e-8 for r in rr)),ordered_recovery=sum(min(r['ordered_cosines'])>.99 for r in rr),matched_recovery=sum(min(r['matched_cosines'])>.99 for r in rr),
 solver_success=sum(r['success'] for r in rr),median_nfev=float(np.median([r['nfev'] for r in rr]))))
result=dict(status='passed',fits=20,quadrature_points=len(x),summary=summary,checks=checks,results_sha256=hashlib.sha256((root/'results.json').read_bytes()).hexdigest())
(root/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary,indent=2));print('Max Jacobian error',max(r['max_relative_jacobian_error'] for r in checks))

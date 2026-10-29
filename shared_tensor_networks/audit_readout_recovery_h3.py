"""Independent polynomial/torch Jacobian and held-out regression replay."""
import json,itertools
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from experiment import gram
from multitask import product_map,monomials
from run_svhn_standard import digest,write
from run_readout_recovery_h3 import maps,D,H,N,P,PAIRS,MASKS

torch.set_num_threads(1)
root=Path(__file__).parent;out=root/'results/readout_recovery_h3'
assert json.loads((out/'completed.json').read_text())['fits']==24
protocol=json.loads((out/'protocol.json').read_text());assert digest(out/'source.py')==protocol['source_sha256']
keys=monomials(D);G=gram(keys);L=np.linalg.cholesky(G).T
records=[];max_j=0.;max_loss=0.
x=np.random.default_rng(217001).normal(size=(31,D));phi=np.prod(x[:,None,:]**np.array(keys)[None,:,:],axis=2)
for path in sorted(out.glob('218*.json')):
 r=json.loads(path.read_text());p=path.with_suffix('.npz');assert digest(p)==r['checkpoint_sha256'];d=np.load(p)
 a,b,w=d['teacher_A'],d['teacher_B'],d['teacher_W'];assert np.linalg.matrix_rank(np.concatenate([a,b]))==D
 assert np.linalg.matrix_rank(w[:P])==P
 teacherK=product_map(a,b,keys);assert np.linalg.matrix_rank(teacherK)==P
 k={'one':1,'three':3,'six':P,'interventions':1}[r['arm']]
 masks=MASKS
 truth=teacherK@(w[:k] if r['arm']!='interventions' else w[:1]*masks).T
 scales=np.sqrt(np.sum(truth*(G@truth),axis=0));np.testing.assert_allclose(scales,d['scales'],atol=1e-12,rtol=1e-12)
 np.testing.assert_allclose(truth/scales,d['teacher_coefficients'],atol=1e-12,rtol=1e-12)
 held=teacherK@w[P:].T;held/=np.sqrt(np.sum(held*(G@held),axis=0));np.testing.assert_allclose(held,d['heldout_coefficients'],atol=1e-12,rtol=1e-12)
 for state in ['initial','final']:
  v=d[state];A=v[:N].reshape(H,D);B=v[N:2*N].reshape(H,D);W=v[2*N:].reshape(-1,P)
  K=product_map(A,B,keys);C=K@(W if r['arm']!='interventions' else W*masks).T
  actual,jac,_,_=maps(v,r['arm']);np.testing.assert_allclose(C,actual,atol=1e-10,rtol=1e-10)
  tx=torch.tensor(x)
  def forward(v):
   A=v[:N].reshape(H,D);B=v[N:2*N].reshape(H,D);W=v[2*N:].reshape(-1,P)
   h=(tx@A.T)*(tx@B.T);products=torch.stack([h[:,i]*h[:,j] for i,j in PAIRS],dim=1)
   if r['arm']=='interventions':W=W*torch.tensor(masks)
   return products@W.T
  auto=torch.func.jacrev(forward)(torch.tensor(v)).numpy();pred=np.einsum('nk,ktp->ntp',phi,jac)
  error=float(np.linalg.norm(auto-pred)/max(1,np.linalg.norm(auto)));assert error<1e-9;max_j=max(max_j,error)
 delta=C/scales-d['teacher_coefficients'];mse=float(np.sum(delta*(G@delta))/len(scales));error=abs(mse-r['mse']);assert error<1e-9;max_loss=max(max_loss,error)
 # Independently evaluate quadratic functions on enough Gaussian-exact nodes.
 z,weights=np.polynomial.hermite_e.hermegauss(3);ids=np.array(list(itertools.product(range(3),repeat=D)));qx=z[ids];qw=np.prod(weights[ids]/np.sqrt(2*np.pi),axis=1)
 tq=(qx@a.T)*(qx@b.T);sq=(qx@A.T)*(qx@B.T);tq*=np.sqrt(qw)[:,None];sq*=np.sqrt(qw)[:,None]
 tu,ts,_=np.linalg.svd(tq,full_matrices=False);su,ss,_=np.linalg.svd(sq,full_matrices=False)
 assert np.sum(ts>ts[0]*1e-11)==np.sum(ss>ss[0]*1e-11)==H
 cos=np.linalg.svd(tu.T@su,compute_uv=False);np.testing.assert_allclose(cos,r['principal_cosines'],atol=1e-9,rtol=1e-9)
 sim=np.abs((tq/np.linalg.norm(tq,axis=0)).T@(sq/np.linalg.norm(sq,axis=0)));ii,jj=linear_sum_assignment(-sim)
 np.testing.assert_allclose(sim[ii,jj],r['component_cosines'],atol=1e-9,rtol=1e-9)
 px=np.prod(d['adaptation_inputs'][:,None,:]**np.array(keys)[None,:,:],axis=2)
 for n in ['population','32','128']:
  read=d['readout_'+n];X,Y=(L@K,L@held) if n=='population' else (px[:int(n)]@K,px[:int(n)]@held)
  assert np.linalg.norm(X.T@(X@read-Y))/max(1,np.linalg.norm(X.T@Y))<1e-8
  delta=K@read-held;loss=np.sum(delta*(G@delta),axis=0);np.testing.assert_allclose(loss,r['transfer_mse'][n],atol=1e-9,rtol=1e-9)
 records.append(dict(name=r['name'],result_sha256=digest(path),checkpoint_sha256=digest(p)))
assert len(records)==24
write(out/'audit.json',dict(status='passed',fits=24,max_relative_jacobian_error=max_j,max_loss_error=max_loss,records=records,source_sha256=digest(Path(__file__))))
print('Audited24',max_j,max_loss)

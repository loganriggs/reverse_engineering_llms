"""Replay finite budgets, independent forward/Jacobian, Gaussian errors and transfer."""
import itertools,json
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from experiment import gram
from multitask import monomials,product_map
from run_svhn_standard import digest,write
from run_readout_recovery_h3 import maps

torch.set_num_threads(1);root=Path(__file__).parent;out=root/'results/finite_readout_recovery'
assert json.loads((out/'completed.json').read_text())['fits']==72
protocol=json.loads((out/'protocol.json').read_text());assert digest(out/'source.py')==protocol['source_sha256']
keys=monomials(6);G=gram(keys);pairs=list(itertools.combinations_with_replacement(range(3),2));records=[];worst_j=0.;worst_loss=0.
def design(x):return np.prod(x[:,None,:]**np.array(keys)[None,:,:],axis=2)
for seed in protocol['teacher_seeds']:
 meta=json.loads((out/f'{seed}_data.json').read_text());assert digest(root/meta['input_file'])==meta['input_sha256'];assert digest(out/f'{seed}_data.npz')==meta['data_sha256']
 data=np.load(out/f'{seed}_data.npz');source=np.load(root/meta['input_file']);rng=np.random.default_rng(seed+5000)
 for name,shape in [('inputs',(1536,6)),('noise',(1536,6)),('adaptation_inputs',(128,6)),('adaptation_noise',(128,4))]:np.testing.assert_array_equal(data[name],rng.normal(size=shape))
 for name in ['teacher_A','teacher_B','teacher_W']:np.testing.assert_array_equal(data[name],source[name])
 oracle=product_map(data['teacher_A'],data['teacher_B'],keys);C=oracle@data['teacher_W'].T;scales=np.sqrt(np.sum(C*(G@C),axis=0));np.testing.assert_allclose(scales,data['norms'],atol=1e-12,rtol=1e-12);truth=C/scales
 np.testing.assert_allclose(truth,data['truth'],atol=1e-12,rtol=1e-12)
 phi=design(data['inputs']);pa=design(data['adaptation_inputs'])
 for path in sorted(out.glob(f'{seed}_b*.json')):
  r=json.loads(path.read_text());d=np.load(path.with_suffix('.npz'));assert digest(path.with_suffix('.npz'))==r['checkpoint_sha256'];m=r['heads'];n=r['ninputs'];sigma=r['noise_std'];assert n*m==r['budget'] and m in [1,3,6]
  y=phi[:n]@truth[:,:m]+sigma*data['noise'][:n,:m]
  expected=np.concatenate([data['teacher_A'].ravel(),data['teacher_B'].ravel(),data['teacher_W'][:m].ravel()]) if r['initialization']=='teacher' else np.concatenate([source['initial'][:36],source['initial'][36:].reshape(6,6)[:m].ravel()])
  np.testing.assert_array_equal(d['initial'],expected)
  for state in ['initial','final']:
   v=d[state];A=v[:18].reshape(3,6);B=v[18:36].reshape(3,6);W=v[36:].reshape(m,6);K=product_map(A,B,keys);co=K@W.T/scales[:m]
   tx=torch.tensor(data['inputs'][:31])
   def forward(v):
    h=(tx@v[:18].reshape(3,6).T)*(tx@v[18:36].reshape(3,6).T);p=torch.stack([h[:,i]*h[:,j] for i,j in pairs],1)
    return p@v[36:].reshape(m,6).T/torch.tensor(scales[:m])/np.sqrt(r['budget'])
   auto=torch.func.jacrev(forward)(torch.tensor(v)).numpy();J=maps(v,'six')[1]
   actual=np.einsum('nk,ktp->ntp',phi[:31],J)/scales[None,:m,None]/np.sqrt(r['budget'])
   error=float(np.linalg.norm(auto-actual)/max(1,np.linalg.norm(auto)));assert error<1e-8;worst_j=max(worst_j,error)
  train=float(np.mean((phi[:n]@co-y)**2));delta=co-truth[:,:m];loss=np.sum(delta*(G@delta),axis=0)
  np.testing.assert_allclose(loss,r['population_mse'],atol=1e-8,rtol=1e-8);assert abs(train-r['train_mse'])<1e-8;worst_loss=max(worst_loss,abs(train-r['train_mse']))
  baseline=d['polynomial_baseline'];normal=phi[:n].T@(phi[:n]@baseline-y);assert np.linalg.norm(normal)/max(1,np.linalg.norm(phi[:n].T@y))<1e-8
  delta=baseline-truth[:,:m];np.testing.assert_allclose(np.sum(delta*(G@delta),axis=0),r['polynomial_population_mse'],atol=1e-8,rtol=1e-8)
  # Independent quadrature verifies component matching (degree4 integrals).
  z,w=np.polynomial.hermite_e.hermegauss(3);ids=np.array(list(itertools.product(range(3),repeat=6)));qx=z[ids];qw=np.prod(w[ids]/np.sqrt(2*np.pi),axis=1)
  tq=np.sqrt(qw)[:,None]*(qx@data['teacher_A'].T)*(qx@data['teacher_B'].T);sq=np.sqrt(qw)[:,None]*(qx@A.T)*(qx@B.T)
  sim=np.abs((tq/np.linalg.norm(tq,axis=0)).T@(sq/np.linalg.norm(sq,axis=0)));ii,jj=linear_sum_assignment(-sim)
  np.testing.assert_allclose(sim[ii,jj],r['component_cosines'],atol=1e-8,rtol=1e-8)
  for count in [32,128]:
   Y=pa[:count]@truth[:,6:]+sigma*data['adaptation_noise'][:count]
   for feature,Kf in [('learned',K),('oracle',oracle)]:
    name=f'{feature}_{count}';read=d['readout_'+name];Z=pa[:count]@Kf
    assert np.linalg.norm(Z.T@(Z@read-Y))/max(1,np.linalg.norm(Z.T@Y))<1e-8
    delta=Kf@read-truth[:,6:];np.testing.assert_allclose(np.sum(delta*(G@delta),axis=0),r['transfer_mse'][name],atol=1e-8,rtol=1e-8)
  records.append(dict(name=r['name'],result_sha256=digest(path),checkpoint_sha256=digest(path.with_suffix('.npz'))))
assert len(records)==72
write(out/'audit.json',dict(status='passed',fits=72,max_relative_jacobian_error=worst_j,max_train_loss_error=worst_loss,records=records,source_sha256=digest(Path(__file__))))
print('Audited72',worst_j,worst_loss)

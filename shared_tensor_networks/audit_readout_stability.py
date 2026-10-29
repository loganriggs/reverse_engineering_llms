"""Independent direct-forward derivatives and polynomial noise/feature replay."""
import json,itertools
from pathlib import Path
import numpy as np
import torch
from experiment import gram
from multitask import product_map,monomials
from run_svhn_standard import digest,write

torch.set_num_threads(1);root=Path(__file__).parent;out=root/'results/readout_stability'
assert json.loads((out/'completed.json').read_text())['fits']==48
protocol=json.loads((out/'protocol.json').read_text());assert digest(out/'source.py')==protocol['source_sha256']
keys=monomials(6);L=np.linalg.cholesky(gram(keys)).T
pairs=list(itertools.combinations_with_replacement(range(3),2));qpairs=list(itertools.combinations_with_replacement(range(6),2));qkeys=[tuple(pair.count(i) for i in range(6)) for pair in qpairs];L2=np.linalg.cholesky(gram(qkeys)).T
masks=np.array([[1]*6]+[[int(i!=h and j!=h) for i,j in pairs] for h in range(3)])
x=np.random.default_rng(221001).normal(size=(31,6));phi=np.prod(x[:,None,:]**np.array(keys)[None,:,:],axis=2)
records=[];worst_j=0.;worst_f=0.;worst_loss=0.
for gp in sorted(out.glob('*_geometry.json')):
 g=json.loads(gp.read_text());assert digest(root/g['input_file'])==g['input_sha256'];gd=np.load(gp.with_suffix('.npz'));v0=gd['teacher'];scales=gd['scales'];m=len(scales);arm=g['name'].split('_',1)[1]
 original=np.load(root/g['input_file']);k={'one':1,'three':3,'six':6,'interventions':1}[arm]
 np.testing.assert_array_equal(v0,np.concatenate([original['teacher_A'].ravel(),original['teacher_B'].ravel(),original['teacher_W'][:k].ravel()]))
 def direct(v):
  A=v[:18].reshape(3,6);B=v[18:36].reshape(3,6);W=v[36:].reshape(-1,6);h=(torch.tensor(x)@A.T)*(torch.tensor(x)@B.T)
  pp=torch.stack([h[:,i]*h[:,j] for i,j in pairs],1)
  if arm=='interventions':W=W*torch.tensor(masks)
  return pp@W.T/torch.tensor(scales)/np.sqrt(m)
 def feature(v):
  A=v[:18].reshape(3,6);B=v[18:36].reshape(3,6)
  q=torch.stack([A[:,i]*B[:,j]+(A[:,j]*B[:,i] if i!=j else 0) for i,j in qpairs]);q=torch.tensor(L2)@q
  return (q/torch.linalg.vector_norm(q,dim=0)/np.sqrt(3)).reshape(-1)
 auto=torch.func.jacrev(direct)(torch.tensor(v0)).numpy()
 rawj=np.linalg.solve(L,gd['J'].reshape(126,-1)).reshape(126,m,len(v0));pred=np.einsum('nk,ktp->ntp',phi,rawj)
 je=float(np.linalg.norm(auto-pred)/max(1,np.linalg.norm(auto)));assert je<1e-9;worst_j=max(worst_j,je)
 F=torch.func.jacrev(feature)(torch.tensor(v0)).numpy();fe=float(np.linalg.norm(F-gd['F']));assert fe<1e-9;worst_f=max(worst_f,fe)
 response=F@np.linalg.pinv(gd['J'],rcond=1e-10);np.testing.assert_allclose(response,gd['response'],atol=1e-8,rtol=1e-8)
 gain=np.linalg.svd(response,compute_uv=False)[0];np.testing.assert_allclose(gain,g['worst_feature_gain'],rtol=1e-10)
 for path in sorted(out.glob(g['name']+'_*.json')):
  if path==gp:continue
  r=json.loads(path.read_text());saved=np.load(path.with_suffix('.npz'));assert digest(path.with_suffix('.npz'))==r['checkpoint_sha256'];np.testing.assert_array_equal(saved['initial'],v0)
  eta=saved['noise'];np.testing.assert_allclose(np.linalg.norm(eta),1,atol=1e-12)
  if r['direction']=='worst':np.testing.assert_allclose(np.linalg.norm(response@eta),gain,rtol=1e-9)
  pred=r['amplitude']*response@eta;actual=(feature(torch.tensor(saved['final']))-feature(torch.tensor(v0))).numpy()
  np.testing.assert_allclose(pred,saved['predicted_feature_change'],atol=1e-10,rtol=1e-8);np.testing.assert_allclose(actual,saved['actual_feature_change'],atol=1e-12,rtol=1e-9)
  def output(v):
   K=product_map(v[:18].reshape(3,6),v[18:36].reshape(3,6),keys);W=v[36:].reshape(-1,6)
   if arm=='interventions':W=W*masks
   return (L@(K@W.T/scales)/np.sqrt(m)).ravel()
  delta=output(saved['final'])-output(v0);clean=float(delta@delta);noisy=float(np.sum((delta-r['amplitude']*eta)**2))
  e=max(abs(clean-r['clean_output_mse']),abs(noisy-r['noisy_fit_mse']));assert e<1e-10;worst_loss=max(worst_loss,e)
  np.testing.assert_allclose([np.linalg.norm(actual),np.linalg.norm(pred),np.linalg.norm(actual-pred)/np.linalg.norm(pred)],[r['feature_change'],r['predicted_change'],r['relative_linearization_error']],atol=1e-8,rtol=1e-7)
  records.append(dict(name=r['name'],result_sha256=digest(path),checkpoint_sha256=digest(path.with_suffix('.npz'))))
assert len(records)==48
write(out/'audit.json',dict(status='passed',fits=48,max_jacobian_error=worst_j,max_feature_derivative_error=worst_f,max_loss_error=worst_loss,records=records,source_sha256=digest(Path(__file__))))
print('Audited48',worst_j,worst_f,worst_loss)

"""M42 independent NumPy PCA/ridge audit and native source replay."""
from pathlib import Path
import json,hashlib
import numpy as np
import torch
from inspect_model import load_model,trace
P=Path(__file__).resolve().parent/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.set_grad_enabled(False)
 rp=P/'native_half_substitution_training.json';r=json.loads(rp.read_text());pp=P/'native_half_substitution_protocol.json';q=json.loads(pp.read_text());assert r['protocol_sha256']==sha(pp);assert sha(P.parent/'train_native_half_substitution.py')==q['source_sha256'];cache=Path(r['cache_path']);assert sha(cache)==r['cache_sha256'];data=torch.load(cache,weights_only=True);assert data['protocol_sha256']==sha(pp);assert r['forward_count']==32
 checks=[]
 for e in r['models']:
  path=Path(e['path']);assert sha(path)==e['sha256'];s=torch.load(path,weights_only=True);assert s['protocol_sha256']==sha(pp);x=data['halves'][f"{e['layer']}:{e['retained']}"].numpy().astype(float);y=data['halves'][f"{e['layer']}:{e['removed']}"].numpy().astype(float);assert x.shape==y.shape==(4096,1152)
  mx=x.mean(0);my=y.mean(0);U=s['U'].numpy();B=s['B'].numpy();assert np.allclose(mx,s['input_mean'].numpy(),rtol=1e-10,atol=1e-10);assert np.allclose(my,s['target_mean'].numpy(),rtol=1e-10,atol=1e-10);xc=x-mx;yc=y-my;cov=xc.T@xc;ev=np.linalg.eigvalsh(cov)[-32:];assert np.allclose(ev,s['eigenvalues'].numpy(),rtol=1e-8,atol=1e-8);assert np.allclose(U.T@U,np.eye(32),atol=1e-10);eigen_res=np.linalg.norm(cov@U-U*ev)/np.linalg.norm(cov@U);assert eigen_res<1e-9
  z=xc@U;lam=.001*np.sum(z*z)/32;assert np.isclose(lam,e['ridge'],rtol=1e-10);aug=np.concatenate([z,np.sqrt(lam)*np.eye(32)]);target=np.concatenate([yc,np.zeros((32,1152))]);ind=np.linalg.lstsq(aug,target,rcond=None)[0];err=np.linalg.norm(ind-B)/np.linalg.norm(B);assert err<1e-9
  me=np.sum(yc**2);fe=np.sum((z@ind-yc)**2);assert np.isclose(me,e['training_mean_error'],rtol=1e-10);assert np.isclose(fe,e['training_map_error'],rtol=1e-10);checks.append(dict(layer=e['layer'],removed=e['removed'],eigen_residual=eigen_res,independent_coefficient_relative_error=err,map_mean_ratio=fe/me))
 assert {(e['layer'],e['removed']) for e in r['models']}=={(l,h) for l in [0,1,2] for h in ['half_01','half_23']}
 # Independently re-read a training document and reproduce both physical native half writes.
 cp=P/'natural_token_context_cohort.json';assert sha(cp)==q['cohort_sha256'];cohort=json.loads(cp.read_text());assert not set(x['document_hash'] for x in cohort['records'][:32])&set(x['document_hash'] for x in cohort['records'][32:]);dp=P/'mlp1_group_deletion.json';assert sha(dp)==q['masks_sha256'];masks=json.loads(dp.read_text())['removed_sets'];model,receipt=load_model();assert receipt.weights_sha256==r['checkpoint_sha256'];dev=next(model.parameters()).device;ids=torch.tensor([cohort['token_ids'][0]],device=dev);_,cap=trace(model,ids)
 for layer in [0,1,2]:
  for half in ['half_01','half_23']:
   mask=torch.zeros(4608,device=dev);mask[masks[half]]=1;part=model.transformer.h[layer].mlp.Down(cap[layer]['products']*mask)[0].cpu();assert torch.equal(part,data['halves'][f'{layer}:{half}'][:128])
 out=dict(status='passed',result_sha256=sha(rp),checks=checks,native_training_document_replayed=0,native_half_captures_exact=6,audit_forwards=1,train_documents=list(range(32)),scope='All six maps independently checked with NumPy eigenspectrum and augmented least squares; native half captures replayed on first training document; document separation checked. No transfer outcome yet.')
 (P/'native_half_substitution_training_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()

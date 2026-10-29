"""Native three-corner arithmetic and gradient connectivity; no optimizer steps."""
import json
import numpy as np
import torch
from bilinear_input_runtime import InputWritePredictor
from bilinear_input_causal_loss import InputCausalExperiment
from fit_pca_encoder import sha,V,A
P=V.parent

def lp(z):
 a=z.detach().cpu().double().numpy();a=a-a.max();return a-np.log(np.exp(a).sum())

def main():
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');pp=V/'bilinear_input_native_loss_check_protocol.json';protocol=json.loads(pp.read_text())
 for name,digest in protocol['references'].items():assert sha(V/name)==digest
 e=InputCausalExperiment();manifest=json.loads((V/'bilinear_input_initial_states.json').read_text());pairs=json.loads((V/'training_causal_coverage_protocol.json').read_text())['pairs'];first=pairs[0]['id'];unequal=next(p['id'] for p in pairs if not p['equal_length']);rows=[];samples=[];maxmetric=0;maxteacher=0
 for entry in manifest['models']:
  assert sha(entry['path'])==entry['sha256'];f=InputWritePredictor(torch.load(entry['path'],weights_only=True)).cuda()
  for pid in [first]+([unequal] if entry['kind']=='dense' and entry['seed']==71 else []):
   for ep in [0,1]:
    key=pid+f'|{ep}';donor=e.donor_key(key);f.zero_grad(set_to_none=True);qr,_,_=f(e.input_cache[key]['input'].cuda());qd,_,_=f(e.input_cache[donor]['input'].cuda());qr.retain_grad();qd.retain_grad();K,B,intact,branches=e.joint_losses(qr,qd,key);c=e.cache[key];nlp=lp(c['native']);kl=float(np.sum(np.exp(nlp)*(nlp-lp(intact))));tol=protocol['loss_tolerances'];assert np.isclose(kl,float(K.detach()),rtol=tol['relative'],atol=tol['absolute']);raw=[]
    for branch in branches:
     t=next(t for t in c['targets'] if t['region']==branch['region']);teacher=lp(t['native_changed'])-nlp;response=lp(branch['changed'])-lp(branch['base']);value=float(np.sum(np.exp(nlp)*(response-teacher)**2));error=abs(value-float(branch['R'].detach()));maxmetric=max(maxmetric,error);assert np.isclose(value,float(branch['R'].detach()),rtol=tol['relative'],atol=tol['absolute'])
     if branch['region']!='identity':raw.append(value/e.den[c['family'],branch['region']])
    assert np.isclose(np.mean(raw),float(B.detach()),rtol=tol['relative'],atol=tol['absolute']);assert len(branches)==(9 if pid==first else 6)
    loss=K+B;loss.backward();assert qr.grad is not None and qd.grad is not None and torch.isfinite(qr.grad).all() and torch.isfinite(qd.grad).all();assert qr.grad.norm()>0 and qd.grad.norm()>0;assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in f.parameters());assert any(bool(p.grad.count_nonzero()) for p in f.parameters());assert all(p.grad is None for p in e.model.parameters())
    rows.append(dict(kind=entry['kind'],seed=entry['seed'],key=key,branches=len(branches),K=float(K.detach()),balanced_response=float(B.detach()),recipient_gradient_norm=float(qr.grad.norm()),donor_gradient_norm=float(qd.grad.norm())))
    samples.append(dict(row=len(rows)-1,intact=intact.detach().cpu(),native=c['native'],branches=[dict(region=b['region'],corner=b['corner'],base=b['base'].detach().cpu(),changed=b['changed'].detach().cpu(),R=float(b['R'].detach())) for b in branches]))
    del loss,K,B,intact,branches,qr,qd
 # Teacher predictions should recover intact and all intervention response targets.
 with torch.no_grad():
  for pid in [first,unequal]:
   for ep in [0,1]:
    key=pid+f'|{ep}';donor=e.donor_key(key);K,B,_,branches=e.joint_losses(e.cache[key]['write'].cuda(),e.cache[donor]['write'].cuda(),key);error=max(float(K),max(float(b['R']) for b in branches));assert error<protocol['native_teacher_recovery_max_loss'];maxteacher=max(maxteacher,error)
 path=A/'emlp_m81_native_loss_samples.pt';assert not path.exists();torch.save(samples,path)
 result=dict(status='passed_connectivity_and_forward_arithmetic',source_sha256=sha(__file__),loss_source_sha256=sha(P/'bilinear_input_causal_loss.py'),runtime_sha256=sha(P/'bilinear_input_runtime.py'),protocol_sha256=sha(pp),rows=rows,max_metric_absolute_error=maxmetric,max_teacher_recovery_loss=maxteacher,forward_count=e.calls,samples_path=str(path),samples_sha256=sha(path),scope='Independent NumPy scoring and autograd connectivity on18fixed training cases; teacher recovery on4cases. Native finite-difference endpoint/parameter checks still pending. No optimizer updates.')
 out=V/'bilinear_input_native_loss_checks.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print('PASSED',len(rows),'cases',e.calls,'forwards',maxmetric,maxteacher)
if __name__=='__main__':main()

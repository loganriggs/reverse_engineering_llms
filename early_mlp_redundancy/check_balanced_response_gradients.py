"""M60 live initial replay, identity and finite-difference preflight; no updates."""
import json,torch,numpy as np
from balanced_response_loss import Experiment,P,V,sha

def main():
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');e=Experiment();checks=[]
 for entry in e.protocol['models']:
  f,mu,scale=e.featurizer(entry);assignment=e.protocol['schedules'][str(entry['seed'])]['task_updates'][0];key=assignment['pair_id']+'|0';c=e.cache[key];write=c['write'].cuda()
  with torch.no_grad():
   pred,_,_=f((write-mu)/scale);q0=pred*scale+mu;ik,ir,ib,_=e.losses(write,c)
  assert max(abs(float(ik)),abs(float(ir)),abs(float(ib)))<1e-9
  expected=next(r for r in e.cal['rows'] if r['kind']==entry['kind'] and r['seed']==entry['seed'] and r['task_update']==0 and r['key']==key)
  for objective,weight in [('raw',e.cal['raw_weight']),('balanced',e.cal['balanced_weight'])]:
   q=q0.detach().requires_grad_();K,R,B,branches=e.losses(q,c)
   for val,name in [(K,'K'),(R,'R_raw'),(B,'R_balanced')]:assert np.isclose(float(val.detach()),expected[name],rtol=1e-8,atol=1e-10)
   value=K+weight*(R if objective=='raw' else B);grad=torch.autograd.grad(value,q)[0];assert torch.isfinite(grad).all() and grad.norm()>0;direction=(grad/grad.norm()*q0.norm()).detach();analytic=float((grad.double()*direction.double()).sum());fds=[]
   for eps in [1e-3,3e-4]:
    with torch.no_grad():
     pk,pr,pb,_=e.losses(q0+eps*direction,c);mk,mr,mb,_=e.losses(q0-eps*direction,c);plus=float(pk+weight*(pr if objective=='raw' else pb));minus=float(mk+weight*(mr if objective=='raw' else mb))
    numeric=(plus-minus)/(2*eps);error=abs(numeric-analytic);relative=error/max(abs(analytic),1e-30);assert relative<.03 or error<1e-6,(entry['kind'],entry['seed'],objective,eps,relative);fds.append(dict(epsilon=eps,analytic=analytic,numeric=numeric,relative_error=relative))
   f.zero_grad();pred,_,_=f((write-mu)/scale);fk,fr,fb,branches=e.losses(pred*scale+mu,c);(fk+weight*(fr if objective=='raw' else fb)).backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in f.parameters());assert all(p.grad is None for p in e.model.parameters());checks.append(dict(kind=entry['kind'],seed=entry['seed'],objective=objective,key=key,identity=[float(ik),float(ir),float(ib)],initial_replay=True,finite_differences=fds,parameter_gradient_norms={n:float(p.grad.norm()) for n,p in f.named_parameters()}));print('CHECKED',entry['kind'],entry['seed'],objective,flush=True)
 out=dict(status='passed',protocol_sha256=sha(e.pp),calibration_sha256=sha(V/'balanced_response_calibration.json'),source_sha256=sha(P/'check_balanced_response_gradients.py'),loss_source_sha256=sha(P/'balanced_response_loss.py'),cache_path=str(e.cachepath),cache_sha256=sha(e.cachepath),checks=checks,forward_count=e.calls,scope='First scheduled endpoint at each of six starts, both causal objectives,24write-space directional finite differences; native identity and exact initial losses; finite selected-support parameter gradients. No optimizer updates or claim of derivatives across hard gate changes.')
 op=V/'balanced_response_gradient_checks.json';assert not op.exists();op.write_text(json.dumps(out,indent=2)+'\n');print('DONE',e.calls)
if __name__=='__main__':main()

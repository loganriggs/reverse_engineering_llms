"""M61 natural KL identity/gradient and fixed-schedule preflight, no fits."""
import json,torch,numpy as np
from balanced_response_loss import Experiment,P,V,sha
from balanced_training_runtime import load_natural,natural_loss

def main():
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');e=Experiment();natural,path=load_natural(e);checks=[]
 for entry in e.protocol['models']:
  schedule=e.protocol['schedules'][str(entry['seed'])];assert len(schedule['natural_documents'])==66 and len(schedule['task_updates'])==66;doc=schedule['natural_documents'][0];c=natural[doc];f,mu,scale=e.featurizer(entry)
  with torch.no_grad():
   identity,out=natural_loss(e,c['write'].cuda(),c);assert torch.equal(out.cpu(),c['native']);assert abs(float(identity))<1e-9;pred,_,_=f((c['write'].cuda()-mu)/scale);q0=pred*scale+mu
  q=q0.detach().requires_grad_();K,out=natural_loss(e,q,c);lp=c['native'].double().numpy();lp=lp-lp.max(-1,keepdims=True);lp-=np.log(np.exp(lp).sum(-1,keepdims=True));op=out.detach().cpu().double().numpy();op-=op.max(-1,keepdims=True);op-=np.log(np.exp(op).sum(-1,keepdims=True));ref=float((np.exp(lp)*(lp-op)).sum(-1).mean());assert np.isclose(ref,float(K.detach()),rtol=1e-8,atol=1e-10);grad=torch.autograd.grad(K,q)[0];assert torch.isfinite(grad).all() and grad.norm()>0;direction=(grad/grad.norm()*q0.norm()).detach();analytic=float((grad.double()*direction.double()).sum());fds=[]
  for eps in [1e-3,3e-4]:
   with torch.no_grad():plus,_=natural_loss(e,q0+eps*direction,c);minus,_=natural_loss(e,q0-eps*direction,c)
   numeric=float((plus-minus)/(2*eps));rel=abs(numeric-analytic)/max(abs(analytic),1e-30);assert rel<.03;fds.append(dict(epsilon=eps,analytic=analytic,numeric=numeric,relative_error=rel))
  f.zero_grad();pred,_,_=f((c['write'].cuda()-mu)/scale);value,_=natural_loss(e,pred*scale+mu,c);value.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in f.parameters());assert all(p.grad is None for p in e.model.parameters());checks.append(dict(kind=entry['kind'],seed=entry['seed'],document=doc,identity=float(identity),KL=float(value.detach()),numpy_KL=ref,finite_differences=fds));print('CHECKED',entry['kind'],entry['seed'],flush=True)
 out=dict(status='passed',protocol_sha256=sha(e.pp),source_sha256=sha(P/'check_balanced_natural.py'),runtime_sha256=sha(P/'balanced_training_runtime.py'),runner_sha256=sha(P/'train_balanced_response.py'),natural_cache_sha256=sha(path),checks=checks,forward_count=e.calls,scope='All-position natural KL vs NumPy, exact native identity,12write directional differences,finite selected-support parameter gradients; no optimizer steps. Runner enforces immutable receipts and per-step schedule/norm/gradient invariants. Resume numerical equivalence not independently replayed.')
 op=V/'balanced_natural_preflight.json';assert not op.exists();op.write_text(json.dumps(out,indent=2)+'\n');print('DONE',e.calls)
if __name__=='__main__':main()

"""M68 full-parameter directional checks with explicit support-change accounting."""
import json,torch
from encoded_change_loss import EncodedExperiment
from balanced_response_loss import P,V,sha

def main():
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');e=EncodedExperiment();cp=V/'encoded_change_calibration.json';cal=json.loads(cp.read_text());checks=[]
 for entry in e.protocol['models']:
  f,mu,scale=e.featurizer(entry);key=e.protocol['schedules'][str(entry['seed'])]['task_updates'][0]['pair_id']+'|0';c=e.cache[key];d=e.cache[e.donor_key(key)];xr=(c['write'].cuda()-mu)/scale;xd=(d['write'].cuda()-mu)/scale
  def value():
   pr,sr,_=f(xr);pd,sd,_=f(xd);K,R,B,_,_=e.encoded_losses(pr*scale+mu,pd*scale+mu,key);return K+cal['encoded_weight']*B,(sr.detach(),sd.detach())
  loss,supports=value();params=list(f.parameters());grads=torch.autograd.grad(loss,params);gn=torch.sqrt(sum(g.double().square().sum() for g in grads));pn=torch.sqrt(sum(p.detach().double().square().sum() for p in params));directions=[(g.double()/gn*pn).to(g.dtype).detach() for g in grads];analytic=sum(float((g.double()*v.double()).sum()) for g,v in zip(grads,directions));originals=[p.detach().clone() for p in params];trials=[]
  with torch.no_grad():
   for eps in [1e-4,3e-5,1e-5]:
    vals=[];changes=[]
    for sign in [1,-1]:
     for p,b,v in zip(params,originals,directions):p.copy_(b+sign*eps*v)
     v,ss=value();vals.append(float(v));changes.append(sum(int((a!=b).sum()) for a,b in zip(ss,supports)))
    numeric=(vals[0]-vals[1])/(2*eps);rel=abs(numeric-analytic)/max(abs(analytic),1e-30);stable=not any(changes)
    if stable:assert rel<.05 or abs(numeric-analytic)<1e-6,(entry['kind'],entry['seed'],eps,rel)
    trials.append(dict(epsilon=eps,analytic=analytic,numeric=numeric,relative_error=rel,changed_support_coordinates=changes,stable_support=stable))
   for p,b in zip(params,originals):p.copy_(b)
  assert any(x['stable_support'] for x in trials),'No stable finite-difference scale among the predeclared three; do not claim parameter derivative verification.'
  # Exact restore and no optimizer update.
  assert all(torch.equal(p,b) for p,b in zip(params,originals));checks.append(dict(kind=entry['kind'],seed=entry['seed'],key=key,trials=trials));print('PARAMETERS',entry['kind'],entry['seed'],flush=True)
 op=V/'encoded_change_parameter_checks.json';assert not op.exists();op.write_text(json.dumps(dict(status='passed',source_sha256=sha(P/'check_encoded_parameter_gradients.py'),loss_source_sha256=sha(P/'encoded_change_loss.py'),calibration_sha256=sha(cp),checks=checks,forward_count=e.calls,scope='All trainable parameters along one gradient direction per six starts. Three fixed epsilons, all crossings recorded; only stable-support trials verify smooth derivative. No silent example/epsilon replacement and exact parameter restore, no optimizer updates.'),indent=2)+'\n');print('DONE',e.calls)
if __name__=='__main__':main()

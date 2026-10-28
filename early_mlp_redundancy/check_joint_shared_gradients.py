"""M35 verify common loss branches and selected-support gradients before fitting."""
import json,math
from pathlib import Path
import torch
from joint_shared_loss import Experiment,P,sha
from collect_response_targets import response_error

def main():
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');e=Experiment();cp=P/'joint_shared_calibration.json';cal=json.loads(cp.read_text());assert cal['protocol_sha256']==sha(e.pp) and cal['loss_source_sha256']==sha(P.parent/'joint_shared_loss.py');weight=cal['response_weight'];checks=[]
 for entry in e.p['initial_models']:
  f,mu,scale=e.featurizer(entry)
  for region in ['prefix','task']:
   assignment=next(x for x in e.p['target_assignments'] if x['seed']==entry['seed'] and e.targets['records'][x['target_indices'][0]]['region']==region)
   indices=assignment['target_indices'];c=e.cache['inputs'][assignment['input_key']]
   with torch.no_grad():
    pred,support,_=f((c['write'].to(e.device)-mu)/scale);q0=pred*scale+mu
    ik,ir,*_=e.losses(c['write'].to(e.device),c,indices)
   assert abs(float(ik))<1e-9 and float(ir)<1e-9
   q=q0.detach().requires_grad_();K,R,out,changed,Rs=e.losses(q,c,indices);value=K+weight*R
   old=next(x for x in cal['rows'] if x['kind']==entry['kind'] and x['seed']==entry['seed'] and x['task_update']==assignment['task_update'] and x['endpoint']==assignment['endpoint'])
   assert math.isclose(float(K.detach()),old['K'],rel_tol=1e-8,abs_tol=1e-10);assert math.isclose(float(R.detach()),old['R'],rel_tol=1e-8,abs_tol=1e-10)
   # Independent leaf copies explicitly check response gradients on BOTH branches.
   leaves=[]
   for i,v in zip(indices,changed):
    a=out[0,-1].detach().requires_grad_();b=v.detach().requires_grad_();r=response_error(c['native'][0,-1].to(e.device),e.cache['targets'][i]['native_changed'].to(e.device),a,b)
    ga,gb=torch.autograd.grad(r,(a,b));assert torch.isfinite(ga).all() and torch.isfinite(gb).all()
    if not e.targets['records'][i]['noop']:assert ga.norm()>0 and gb.norm()>0
    leaves.append(dict(target_index=i,noop=e.targets['records'][i]['noop'],intact_gradient_norm=float(ga.norm()),changed_gradient_norm=float(gb.norm())))
   grad=torch.autograd.grad(value,q)[0];assert torch.isfinite(grad).all() and grad.norm()>0;direction=(grad/grad.norm()*q0.norm()).detach();analytic=float((grad.double()*direction.double()).sum());fds=[]
   for eps in [1e-3,3e-4]:
    with torch.no_grad():
     pk,pr,*_=e.losses(q0+eps*direction,c,indices);mk,mr,*_=e.losses(q0-eps*direction,c,indices);plus=float(pk+weight*pr);minus=float(mk+weight*mr)
    numeric=(plus-minus)/(2*eps);absolute=abs(numeric-analytic);relative=absolute/max(abs(analytic),1e-30);assert relative<.03 or absolute<1e-6,(entry['kind'],entry['seed'],region,eps,relative)
    fds.append(dict(epsilon=eps,plus=plus,minus=minus,analytic=analytic,numeric=numeric,absolute_error=absolute,relative_error=relative))
   f.zero_grad();pred,_,_=f((c['write'].to(e.device)-mu)/scale);fk,fr,*_=e.losses(pred*scale+mu,c,indices);(fk+weight*fr).backward();norms={name:float(p.grad.norm()) for name,p in f.named_parameters()};assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in f.parameters());assert all(p.grad is None for p in e.model.parameters())
   checks.append(dict(kind=entry['kind'],seed=entry['seed'],region=region,assignment=assignment,identity_K=float(ik),identity_R=float(ir),K=float(K.detach()),R=float(R.detach()),branch_checks=leaves,finite_differences=fds,featurizer_gradient_norms=norms));print('CHECKED',entry['kind'],entry['seed'],region,flush=True)
 result=dict(status='passed',protocol_sha256=sha(e.pp),calibration_sha256=sha(cp),source_sha256=sha(Path(__file__)),loss_source_sha256=sha(P.parent/'joint_shared_loss.py'),response_weight=weight,checks=checks,forward_count=e.calls,scope='First scheduled example per region for all six starts.24 directional finite differences of combined loss in reconstructed write space, both response branches and finite selected-support parameter gradients. Does not claim derivative across hard support changes or optimization success.')
 (P/'joint_shared_gradient_checks.json').write_text(json.dumps(result,indent=2)+'\n');print('DONE',e.calls,flush=True)
if __name__=='__main__':main()

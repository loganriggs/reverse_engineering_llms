"""M68 gradients through both encoded endpoints; fixed examples, no fitting."""
import json,torch
from encoded_change_loss import EncodedExperiment
from balanced_response_loss import P,V,sha

def main():
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');e=EncodedExperiment();cp=V/'encoded_change_calibration.json';cal=json.loads(cp.read_text());assert cal['loss_source_sha256']==sha(P/'encoded_change_loss.py');checks=[]
 for entry in e.protocol['models']:
  f,mu,scale=e.featurizer(entry);key=e.protocol['schedules'][str(entry['seed'])]['task_updates'][0]['pair_id']+'|0';c=e.cache[key];d=e.cache[e.donor_key(key)]
  with torch.no_grad():qr0=f((c['write'].cuda()-mu)/scale)[0]*scale+mu;qd0=f((d['write'].cuda()-mu)/scale)[0]*scale+mu
  qr=qr0.detach().requires_grad_();qd=qd0.detach().requires_grad_();K,R,B,z,branches=e.encoded_losses(qr,qd,key);expected=next(x for x in cal['rows'] if x['kind']==entry['kind'] and x['seed']==entry['seed'] and x['task_update']==0 and x['key']==key);assert float(K)==expected['K'] and float(B)==expected['R_balanced'];fds=[]
  for region in ['changed_input_positions','final_position']:
   target=next(b['R'] for b in branches if b['region']==region);gr,gd=torch.autograd.grad(target,[qr,qd],retain_graph=True);assert torch.isfinite(gr).all() and torch.isfinite(gd).all() and gr.norm()>0 and gd.norm()>0;dr=(gr/gr.norm()*qr0.norm()).detach();dd=(gd/gd.norm()*qd0.norm()).detach();analytic=float((gr.double()*dr.double()).sum()+(gd.double()*dd.double()).sum())
   for eps in [1e-3,3e-4]:
    with torch.no_grad():
     pb=e.encoded_losses(qr0+eps*dr,qd0+eps*dd,key)[4];mb=e.encoded_losses(qr0-eps*dr,qd0-eps*dd,key)[4];plus=float(next(b['R'] for b in pb if b['region']==region));minus=float(next(b['R'] for b in mb if b['region']==region))
    numeric=(plus-minus)/(2*eps);rel=abs(numeric-analytic)/max(abs(analytic),1e-30);assert rel<.03 or abs(numeric-analytic)<1e-6,(entry['kind'],entry['seed'],region,eps,rel);fds.append(dict(region=region,epsilon=eps,analytic=analytic,numeric=numeric,relative_error=rel,recipient_gradient_norm=float(gr.norm()),donor_gradient_norm=float(gd.norm())))
  f.zero_grad();qr=f((c['write'].cuda()-mu)/scale)[0]*scale+mu;qd=f((d['write'].cuda()-mu)/scale)[0]*scale+mu;K,R,B,z,branches=e.encoded_losses(qr,qd,key);gr,gd=torch.autograd.grad(B,[qr,qd],retain_graph=True);assert gr.norm()>0 and gd.norm()>0;(K+cal['encoded_weight']*B).backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in f.parameters());assert all(p.grad is None for p in e.model.parameters())
  checks.append(dict(kind=entry['kind'],seed=entry['seed'],key=key,finite_differences=fds,recipient_encoded_gradient_norm=float(gr.norm()),donor_encoded_gradient_norm=float(gd.norm()),parameter_gradient_norms={n:float(p.grad.norm()) for n,p in f.named_parameters()},scope='Write-space finite differences bypass hard selection; full encoder/decoder parameter graph checked for finite gradients, not cross-boundary differentiability.'));print('GRADIENTS',entry['kind'],entry['seed'],flush=True)
 op=V/'encoded_change_gradient_checks.json';assert not op.exists();op.write_text(json.dumps(dict(status='passed',source_sha256=sha(P/'check_encoded_change_gradients.py'),loss_source_sha256=sha(P/'encoded_change_loss.py'),calibration_sha256=sha(cp),checks=checks,forward_count=e.calls,scope='First scheduled endpoint of each six fixed starts; source/final donor+recipient directional derivatives,24 finite differences; all selected-support parameter gradients finite; native model frozen. No optimizer updates or support-boundary derivative claim.'),indent=2)+'\n');print('DONE',e.calls,flush=True)
if __name__=='__main__':main()

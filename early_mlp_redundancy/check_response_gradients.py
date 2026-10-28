"""M19 finite-difference validation of both branches of the response objective."""
from pathlib import Path
import json,hashlib,math
import torch
from inspect_model import load_model,trace
from train_write_featurizers import Featurizer
from collect_response_targets import response_error
P=Path(__file__).resolve().parent/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');pp=P/'response_supervision_protocol.json';p=json.loads(pp.read_text());tp=P/'response_training_targets.json';t=json.loads(tp.read_text());assert sha(pp)==t['protocol_sha256'];cp=Path(t['cache_path']);assert sha(cp)==t['cache_sha256'];cache=torch.load(cp,weights_only=True);model,receipt=load_model();model.requires_grad_(False);device=next(model.parameters()).device;mlp=model.transformer.h[1].mlp;checks=[]
 for entry in p['initial_models']:
  s=torch.load(entry['path'],weights_only=True)
  for region in ['prefix','task']:
   index=next(i for i,r in enumerate(t['records']) if (r['kind'],r['seed'],r['region'])==(entry['kind'],entry['seed'],region) and not r['noop'] and r['initial_R']>0);row=t['records'][index];data=cache['inputs'][row['input_key']];initial=cache['initials'][row['initial_key']];target=cache['targets'][index];ids=data['ids'].to(device);native=data['native'].to(device)[0,-1].detach();native_changed=target['native_changed'].to(device).detach();delta=target['delta'].to(device).detach();assert not delta.requires_grad
   def outputs(q):
    a,_=trace.__wrapped__(model,ids,replace=lambda i,z,w:q+mlp.Down_bias if i==1 else w);b,_=trace.__wrapped__(model,ids,replace=lambda i,z,w:q-delta+mlp.Down_bias if i==1 else w);return a[0,-1],b[0,-1]
   def loss(q):
    a,b=outputs(q);return response_error(native,native_changed,a,b)
   with torch.no_grad():identity=float(loss(data['write'].to(device)))
   assert identity<1e-9
   q=initial['q'].to(device).detach().requires_grad_();a,b=outputs(q);value=response_error(native,native_changed,a,b);assert math.isclose(float(value.detach()),row['initial_R'],rel_tol=1e-7,abs_tol=1e-10);branches=torch.autograd.grad(value,(a,b),retain_graph=True);assert all(torch.isfinite(g).all() and float(g.norm())>0 for g in branches);grad=torch.autograd.grad(value,q)[0];assert torch.isfinite(grad).all() and float(grad.norm())>0;direction=(grad/grad.norm()*q.detach().norm()).detach();analytic=float((grad.double()*direction.double()).sum());fd=[]
   for eps in [1e-3,3e-4]:
    with torch.no_grad():plus=float(loss(q.detach()+eps*direction));minus=float(loss(q.detach()-eps*direction))
    numeric=(plus-minus)/(2*eps);absolute=abs(numeric-analytic);relative=absolute/max(abs(analytic),1e-30);assert relative<.03 or absolute<1e-6;fd.append(dict(epsilon=eps,analytic=analytic,finite_difference=numeric,relative_error=relative,absolute_error=absolute))
   f=Featurizer(entry['kind'],entry['seed']).to(device);f.load_state_dict(s['state']);mu=s['mean'].to(device);scale=s['scale'].to(device);rec,_,_=f((data['write'].to(device)-mu)/scale);fv=loss(rec*scale+mu);fv.backward();norms={n:float(v.grad.norm()) for n,v in f.named_parameters()};assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in f.parameters());assert sum(norms.values())>0;assert all(v.grad is None for v in model.parameters());checks.append(dict(kind=entry['kind'],seed=entry['seed'],region=region,target_index=index,initial_response_loss=float(value.detach()),identity_response_error=identity,branch_gradient_norms=[float(g.norm()) for g in branches],finite_differences=fd,featurizer_gradient_norms=norms));print('GRADIENT',entry['kind'],entry['seed'],region,'PASS',flush=True)
 out=dict(status='passed',protocol_sha256=sha(pp),targets_sha256=sha(tp),source_sha256=sha(Path(__file__)),checks=checks,scope='Eight train-only selected nonzero response examples; both downstream branches differentiated, identity and16directional finite differences, finite selected-support featurizer gradients. No derivative claim across TopK index changes; no full optimization performed.')
 (P/'response_gradient_checks.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()

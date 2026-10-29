"""Native-model finite differences for endpoint and input-feature parameters."""
import json
import numpy as np
import torch
from bilinear_input_runtime import InputWritePredictor
from bilinear_input_causal_loss import InputCausalExperiment
from balanced_training_runtime import load_natural,natural_loss
from fit_pca_encoder import sha,V,A
P=V.parent

def main():
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');pp=V/'bilinear_input_native_gradient_protocol.json';protocol=json.loads(pp.read_text())
 for name,digest in protocol['references'].items():assert sha(V/name)==digest
 e=InputCausalExperiment();nat,_=load_natural(e);ref=json.loads((V/'native_input_reference.json').read_text());assert sha(ref['path'])==ref['sha256'];ns=torch.load(ref['path'],weights_only=True);zn=ns['z'][:128].cuda()[None];key=json.loads((V/'training_causal_coverage_protocol.json').read_text())['pairs'][0]['id']+'|0';donor=e.donor_key(key);zr=e.input_cache[key]['input'].cuda();zd=e.input_cache[donor]['input'].cuda();manifest=json.loads((V/'bilinear_input_initial_states.json').read_text());rows=[];maxkl=0
 def task(qr,qd):
  K,B,_,_=e.joint_losses(qr,qd,key);return K+B
 def record(entry,scope,ad,vals):
  rows.append(dict(kind=entry['kind'],seed=entry['seed'],scope=scope,autograd=ad,steps=vals,passed=sum(v['support_stable'] and v['agreement'] for v in vals)>=2))
 for entry in manifest['models']:
  assert sha(entry['path'])==entry['sha256'];f=InputWritePredictor(torch.load(entry['path'],weights_only=True)).cuda()
  with torch.no_grad():qr=f(zr)[0];qd=f(zd)[0]
  qr=qr.detach().requires_grad_();qd=qd.detach().requires_grad_();loss=task(qr,qd);grads=torch.autograd.grad(loss,(qr,qd));base=[qr.detach(),qd.detach()]
  for which,scope in [(0,'recipient_output'),(1,'donor_output')]:
   grad=grads[which];norm=grad.norm();assert norm>0 and torch.isfinite(grad).all();direction=grad/norm;ad=float((grad*direction).sum());vals=[]
   for h in protocol['endpoint_steps']:
    values=[]
    with torch.no_grad():
     for sign in [-1,1]:
      args=list(base);args[which]=base[which]+sign*h*direction;values.append(float(task(*args)))
    fd=(values[1]-values[0])/(2*h);vals.append(dict(step=h,finite_difference=fd,absolute_error=abs(fd-ad),support_stable=True,agreement=abs(fd-ad)<=.001+.05*abs(ad)))
   record(entry,scope,ad,vals)
  del loss,grads,qr,qd
  params=list(f.parameters());original=[p.detach().clone() for p in params]
  for scope in ['task_parameters','natural_parameters']:
   f.zero_grad(set_to_none=True)
   if scope=='task_parameters':
    qr,sr,_=f(zr);qd,sd,_=f(zd);loss=task(qr,qd);supports=[sr,sd]
   else:
    q,sr,_=f(zn);loss,logits=natural_loss(e,q,nat[0]);supports=[sr]
    # Independent NumPy all-position KL arithmetic, no float32 metric shortcut.
    def logp(t):
     a=t.detach().cpu().double().numpy();a=a-a.max(-1,keepdims=True);return a-np.log(np.exp(a).sum(-1,keepdims=True))
    p=logp(nat[0]['native']);v=logp(logits);independent=float((np.exp(p)*(p-v)).sum(-1).mean());err=abs(independent-float(loss.detach()));assert np.isclose(independent,float(loss.detach()),rtol=1e-7,atol=1e-9);maxkl=max(maxkl,err)
   gs=torch.autograd.grad(loss,params);norm=torch.sqrt(sum(g.square().sum() for g in gs));assert norm>0 and all(torch.isfinite(g).all() for g in gs);directions=[g/norm for g in gs];ad=float(sum((g*v).sum() for g,v in zip(gs,directions)));vals=[];del loss
   for h in protocol['parameter_steps']:
    values=[];stable=True
    with torch.no_grad():
     for sign in [-1,1]:
      for p,start,d in zip(params,original,directions):p.copy_(start+sign*h*d)
      if scope=='task_parameters':
       qr,s1,_=f(zr);qd,s2,_=f(zd);value=task(qr,qd);stable=stable and torch.equal(s1,supports[0]) and torch.equal(s2,supports[1])
      else:
       q,s1,_=f(zn);value,_=natural_loss(e,q,nat[0]);stable=stable and torch.equal(s1,supports[0])
      values.append(float(value))
     for p,start in zip(params,original):p.copy_(start)
    fd=(values[1]-values[0])/(2*h);vals.append(dict(step=h,finite_difference=fd,absolute_error=abs(fd-ad),support_stable=stable,agreement=abs(fd-ad)<=.001+.05*abs(ad)))
   record(entry,scope,ad,vals)
  assert all(p.grad is None for p in e.model.parameters());print('MODEL',entry['kind'],entry['seed'],'PASS',all(r['passed'] for r in rows[-4:]),flush=True)
 out=dict(status='passed' if all(r['passed'] for r in rows) else 'unresolved',source_sha256=sha(__file__),loss_source_sha256=sha(P/'bilinear_input_causal_loss.py'),runtime_sha256=sha(P/'bilinear_input_runtime.py'),protocol_sha256=sha(pp),forward_count=e.calls,rows=rows,natural_KL_max_independent_error=maxkl,scope='32fixed directions/96steps, training data only; every gate crossing retained. Gradient-aligned checks exercise local native derivatives, not global optimization quality or support-boundary differentiability. No optimizer update.')
 path=V/'bilinear_input_native_gradient_checks.json';assert not path.exists();path.write_text(json.dumps(out,indent=2)+'\n');print(out['status'],e.calls)
if __name__=='__main__':main()

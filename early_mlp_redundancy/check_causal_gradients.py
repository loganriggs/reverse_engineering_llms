"""M8 prerequisite: differentiate native downstream KL through an MLP1 write."""
import json,hashlib
from pathlib import Path
import torch
from inspect_model import load_model,trace
from train_write_featurizers import Featurizer
P=Path(__file__).resolve().parent;A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition')
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');model,receipt=load_model();model.requires_grad_(False);device=next(model.parameters()).device;cohort=json.loads((P/'validation/natural_token_context_cohort.json').read_text());ids=torch.tensor([cohort['token_ids'][0]],device=device);ref,cap=trace(model,ids);lp=ref.double().log_softmax(-1).detach();raw=model.transformer.h[1].mlp.Down(cap[1]['products']).detach();bias=model.transformer.h[1].mlp.Down_bias;state=torch.load(A/'emlp_m7_write_featurizers/scalar_71.pt',weights_only=True);m=Featurizer('scalar',71).to(device);m.load_state_dict(state['state']);mean=state['mean'].to(device);scale=state['scale'].to(device);x=((raw-mean)/scale).detach()
 def logits(q):return trace.__wrapped__(model,ids,replace=lambda i,z,w:q+bias if i==1 else w)[0]
 def loss(q):
  out=logits(q);return (lp.exp()*(lp-out.double().log_softmax(-1))).sum(-1).mean()
 with torch.no_grad():
  identity=logits(raw);identityerr=float((identity-ref).abs().max());assert identityerr==0
  q=m(x)[0]*scale+mean
 q=q.detach().requires_grad_();value=loss(q);grad=torch.autograd.grad(value,q)[0];assert torch.isfinite(grad).all() and float(grad.norm())>0;direction=grad/grad.norm()*q.detach().norm();analytic=float((grad.double()*direction.double()).sum());checks=[]
 for eps in [.0003,.001]:
  with torch.no_grad():plus=float(loss(q+eps*direction));minus=float(loss(q-eps*direction))
  fd=(plus-minus)/(2*eps);relative=abs(fd-analytic)/max(abs(analytic),1e-12);checks.append(dict(epsilon=eps,analytic=analytic,finite_difference=fd,relative_error=relative));assert relative<.02
 pred=m(x)[0]*scale+mean;objective=loss(pred);objective.backward();norms={name:float(p.grad.norm()) if p.grad is not None else None for name,p in m.named_parameters()};assert all(v is not None and v>0 for v in norms.values());assert all(p.grad is None for p in model.parameters())
 out=dict(status='passed',training_document=0,identity_max_logit_error=identityerr,KL=float(value.detach()),finite_difference_checks=checks,featurizer_gradient_norms=norms,native_model_frozen=True,checkpoint_sha256=receipt.weights_sha256,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Selected-support gradients through native downstream computation. TopK index changes remain nondifferentiable; no claim of exact gradient through gate selection.')
 (P/'validation/causal_gradient_check.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()

"""H1046 training-only local surrogate derivative and hard descent diagnostic."""
from pathlib import Path
import sys,json,hashlib,ast,numpy as np,torch
from torch import nn
R=Path('/home/loganriggs/Coding/neuron_circuits');A=R/'artifacts/bilin18_tensor_decomposition';S='bilin18_gate_surrogate_audit';out=A/'gate_surrogate_audit';out.mkdir(exist_ok=True)
repo=Path('/home/loganriggs/Coding/reverse_engineer/tensor_language')
for p in [repo,repo/'basis_aligned/polynomial_causal',repo/'basis_aligned/bilinear_quotient/ops']:sys.path.insert(0,str(p))
import bilin18_observed_model_facade as facade
F=torch.nn.functional;torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');seed=10331
panel=json.loads((A/'training_coverage_selector_comparison/panel.json').read_text())['train'];families=sorted({r['family'] for r in panel});pairs=[next(r for r in panel if r['family']==f) for f in families];assert len(pairs)==10
old=torch.load(A/'multitask_boundary_contract/states.pt',weights_only=False);added=torch.load(A/'diverse_task_expansion_screen/states.pt',weights_only=False);states={}
for p in pairs:
 for i in p['indices']:
  prefix,idx=i.split(':');data=old if prefix=='old' else added;states[i]=tuple(v.cuda() for v in data['states'][int(idx)])
geometry=A/'training_coverage_selector_comparison/expanded_pca.pt';source=R/'scripts/run_bilin18_direct_change_supervision.py';definition=R/'scripts/run_bilin18_finite_supervised_selectors.py';cp=A/'direct_change_supervision';files=[geometry,source,definition]+[cp/f'{k}.pt' for k in ['scalar','block','shared']];hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files};pca=torch.load(geometry,weights_only=False);B=[b.cuda() for b in pca['bases']];scale=pca['scale'].cuda()
config=dict(experiment='H1046',pairs=[p['id'] for p in pairs],selection='First listed training pair in each of10families, fixed before diagnostics. No evaluation datasets loaded.',checkpoints=['initial','final160'],hard_steps=[.001,.01,.03,.1],direction='Negative surrogate gradient normalized to max absolute coordinate1; also first-step Adam-like sign gradient at0.03.',derivative='At initialization compare STE autograd with central differences of continuous local extension hard(w0)+soft(w)-soft(w0). Not finite differences of discontinuous hard selection.',finite_difference_steps=[.001,.01],limits='Local diagnostic, not optimizer tuning or acceptance training. Zero hard change in a constant-support cell is expected. A passed surrogate check is not evidence its direction descends hard loss.',source_hashes=hashes)
(R/f'configs/{S}.json').write_text(json.dumps(config,indent=2)+'\n')
for path,names in [(definition,['Selector','suffix','KL']),(source,['direct'])]:
 tree=ast.parse(path.read_text());nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names];exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),globals())
counts={'suffix':0};model,_=facade.load_bilin18(snapshot=Path('/home/loganriggs/.cache/huggingface/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240'),device='cuda',dtype=torch.float32,verify_weights_sha256=True)
for v in model.parameters():v.requires_grad_(False)
with torch.no_grad():native={i:suffix(st).detach() for i,st in states.items()}
def continuous_support(selector,delta):
 coeff=[];softs=[]
 for k in range(3):
  c=(delta[k].reshape(-1,1152)/scale[k])@B[k];e=c.square();scores=e*selector.importance[k].clamp(-2,2).exp()/(e.mean(1,keepdim=True).detach()+1e-8)
  if selector.kind=='scalar':soft=scores.softmax(1)*16
  else:
   M=selector.membership[k];soft=((scores@M.T/(4 if selector.kind=='block' else 8)).softmax(1)*(4 if selector.kind=='block' else 2))@M
  coeff.append(c);softs.append(soft)
 return coeff,softs
def losses(selector,reference=None):
 vals=[];supports=[]
 for n,p in enumerate(pairs):
  i,d=p['indices'];delta=tuple(states[d][k]-states[i][k] for k in range(3))
  if reference is None:change,ss=direct(selector,delta)
  else:
   cc,soft=continuous_support(selector,delta);hard0,soft0=reference[n];ss=[hard0[k]+soft[k]-soft0[k] for k in range(3)];change=tuple(((cc[k]*ss[k])@B[k].T*scale[k]).reshape_as(delta[k]) for k in range(3))
  yi=suffix(tuple(states[i][k]+change[k] for k in range(3)));yd=suffix(tuple(states[d][k]-change[k] for k in range(3)));vals.append((KL(native[d],yi)+KL(native[i],yd))/2);supports.append([s.detach().clone() for s in ss])
 return torch.stack(vals),supports
records=[];gradchecks=[];max_anchor=0.
for kind in ['scalar','block','shared']:
 for checkpoint in ['initial','final160']:
  selector=Selector(kind)
  if checkpoint=='final160':selector.load_state_dict(torch.load(cp/f'{kind}.pt',weights_only=False)['state'])
  w0=selector.importance.detach().clone();val,base_supports=losses(selector);base=float(val.mean().detach());val.mean().backward();grad=selector.importance.grad.detach().clone();direction=-grad/(grad.abs().max()+1e-12);predicted=float((grad*direction).sum());assert torch.isfinite(grad).all()
  if checkpoint=='initial':
   reference=[]
   with torch.no_grad():
    for n,p in enumerate(pairs):
     i,d=p['indices'];delta=tuple(states[d][k]-states[i][k] for k in range(3));_,soft=continuous_support(selector,delta);reference.append((base_supports[n],[x.detach().clone() for x in soft]))
    anchor,_=losses(selector,reference);max_anchor=max(max_anchor,abs(float(anchor.mean())-base))
    for eps in [.001,.01]:
     selector.importance.copy_(w0+eps*direction);plus,_=losses(selector,reference);selector.importance.copy_(w0-eps*direction);minus,_=losses(selector,reference);fd=float((plus.mean()-minus.mean())/(2*eps));gradchecks.append(dict(kind=kind,epsilon=eps,predicted_derivative=predicted,finite_difference=fd,relative_error=abs(fd-predicted)/max(abs(predicted),1e-8)))
    selector.importance.copy_(w0)
  with torch.no_grad():
   for mode,step in [('gradient',v) for v in [.001,.01,.03,.1]]+[('adam_sign',.03)]:
    update=direction if mode=='gradient' else -grad.sign();proposal=(w0+step*update).clamp(-2,2);selector.importance.copy_(proposal);actual,ss=losses(selector);flips=sum(int((a!=b).sum()) for aa,bb in zip(ss,base_supports) for a,b in zip(aa,bb));records.append(dict(kind=kind,checkpoint=checkpoint,direction=mode,step=step,base_loss=base,actual_loss=float(actual.mean()),actual_change=float(actual.mean())-base,predicted_change=float((grad*(proposal-w0)).sum()),support_bit_flips=flips,per_family_before=val.detach().cpu().tolist(),per_family_after=actual.cpu().tolist()));selector.importance.copy_(w0)
  print('AUDITED',kind,checkpoint,flush=True)
audit=dict(source_hashes_unchanged=all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p)] for p in files),local_surrogate_anchor_max=max_anchor,suffix_sequences=counts['suffix'],evaluation_rows_loaded=0,fit_updates_applied=0,finite=all(np.isfinite(r['actual_loss']) for r in records))
assert audit['source_hashes_unchanged'] and audit['finite'] and max_anchor<1e-5
(A/f'{S}.json').write_text(json.dumps(dict(config=config,records=records,gradient_checks=gradchecks,audits=audit),indent=2)+'\n');print(json.dumps(dict(gradient_checks=gradchecks,audits=audit),indent=2))

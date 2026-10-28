"""H1033 frozen PCA geometry, equal-budget selectors, actual finite downstream loss."""
from pathlib import Path
import sys,json,hashlib,time,numpy as np,torch
from torch import nn
R=Path('/home/loganriggs/Coding/neuron_circuits');A=R/'artifacts/bilin18_tensor_decomposition';S='bilin18_finite_supervised_selectors';out=A/'finite_supervised_selectors';out.mkdir(exist_ok=True)
repo=Path('/home/loganriggs/Coding/reverse_engineer/tensor_language')
for p in [repo,repo/'basis_aligned/polynomial_causal',repo/'basis_aligned/bilinear_quotient/ops']:sys.path.insert(0,str(p))
import bilin18_observed_model_facade as facade
torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');F=torch.nn.functional
trainfile=A/'multitask_boundary_contract/states.pt';pcafile=A/'multistream_span_diagnosis/pca.pt';hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [trainfile,pcafile]};td=torch.load(trainfile,weights_only=False);tp=json.loads((A/'multitask_boundary_contract/panel.json').read_text());dd=torch.load(A/'multistream_sparse_baselines/development_states.pt',weights_only=False);dp=json.loads((A/'multistream_sparse_baselines/development_panel.json').read_text());pca=torch.load(pcafile,weights_only=False);families=sorted({p['family'] for p in tp});train={i:tuple(v.cuda() for v in st) for i,st in td['states'].items()};dev={i:tuple(v.cuda() for v in st) for i,st in dd['states'].items()};native_dev={i:v.cuda() for i,v in dd['native'].items()};B=[b.float().cuda() for b in pca['bases']];mu=pca['center'].float().cuda();scale=pca['scale'].float().cuda();kinds=['scalar','block','shared'];steps=80;seed=10331
schedule=[]
for step in range(steps):
 f=families[step%len(families)];pp=[p for p in tp if p['family']==f];schedule.append(pp[(step//len(families))%len(pp)]['id'])
config={'experiment':'H1033','seed':seed,'steps':steps,'dictionary':'Frozen train-only PCA128 per stream from H1032;48active maximum combined per token; same native port/position access for every method.','models':'384learned log-importance scalars, one per PCA coordinate/port. ScalarTop16; block32x4 Top4groups; shared32x8 Top2groups degree2 fixed. Coordinate values remain exact PCA projections.','optimization':'Adam0.03, fixed80pair steps, balanced round-robin family schedule, pair order cycles through all eligible pairs. Final step only. Log-importance clamped[-2,2];1e-4mean-square penalty.','objectives':{'activation':'mean normalized whole-port MSE +0.1mean normalized pair-difference error','finite_causal':'mean KL(native endpoint || reconstructed endpoint output) + mean KL(full donor || recipient+decoded donor-minus-recipient output), both directions. Native suffix differentiable, weights frozen.'},'gradient':'Hard exact support forward; softmax-based straight-through gate gradients. Surrogate gradient, not derivative of discrete top-k. Fixed temperature1 and normalized score scale.','controls':'Untrained selectors evaluated first; fixed initialization/memberships/schedule/objective-independent evaluation. No architecture or hyperparameter selection. Single seed pilot.','data':'120discovery endpoint occurrences,38historically opened lexical development endpoints, no original whole-task holdouts. Dictionary already fitted on discovery only. Capital diagnostic subset retained.','source_hashes':hashes,'schedule':schedule};(R/f'configs/{S}.json').write_text(json.dumps(config,indent=2)+'\n')
model,_=facade.load_bilin18(snapshot=Path('/home/loganriggs/.cache/huggingface/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240'),device='cuda',dtype=torch.float32,verify_weights_sha256=True)
for p in model.parameters():p.requires_grad_(False)
counts={'suffix':0,'training_student':0}
def suffix(st):
 counts['suffix']+=1;x,x0,bus=st
 for l in range(11,18):
  b=model.transformer.h[l]
  if l>11:x=b.lambdas[0]*x+b.lambdas[1]*x0
  at,_=b.attn(F.rms_norm(x,(1152,)),bus);x=x+at;x=x+b.mlp(F.rms_norm(x,(1152,)))
 return (30*torch.tanh(model.lm_head(F.rms_norm(x,(1152,))[:,-1])/30))[0]
with torch.no_grad():native_train={i:suffix(st).detach() for i,st in train.items()}
def KL(target,y):
 lp=F.log_softmax(target,0);return (lp.exp()*(lp-F.log_softmax(y,0))).sum()
class Selector(nn.Module):
 def __init__(self,kind):
  super().__init__();self.kind=kind;self.importance=nn.Parameter(torch.zeros(3,128,device='cuda'));ms=[]
  for port in range(3):
   M=torch.zeros(32,128,device='cuda');gen=torch.Generator().manual_seed(seed+port)
   if kind=='block':M[torch.arange(32)[:,None],torch.randperm(128,generator=gen).reshape(32,4)]=1
   else:
    for half in range(2):M[torch.arange(16)[:,None]+16*half,torch.randperm(128,generator=gen).reshape(16,8)]=1
   ms.append(M)
  self.register_buffer('membership',torch.stack(ms))
 def forward(self,st):
  parts=[];supports=[]
  for k in range(3):
   raw=st[k].reshape(-1,1152);z=(raw-mu[k])/scale[k];c=z@B[k];energy=c.square();scores=energy*self.importance[k].clamp(-2,2).exp();scores=scores/(energy.mean(1,keepdim=True).detach()+1e-8)
   if self.kind=='scalar':gscore=scores;budget=16
   else:gscore=scores@self.membership[k].T/(4 if self.kind=='block' else 8);budget=4 if self.kind=='block' else 2
   hard=torch.zeros_like(gscore).scatter_(1,gscore.topk(budget,1).indices,1);soft=gscore.softmax(1)*budget;gates=hard+(soft-soft.detach())
   if self.kind=='scalar':support=gates
   else:
    # Exact OR forward with a stable additive straight-through surrogate.
    hs=(hard@self.membership[k]>0).to(c);ss=gates@self.membership[k];support=hs+(ss-ss.detach())
   parts.append(((c*support)@B[k].T*scale[k]+mu[k]).reshape_as(st[k]));supports.append(support)
  return tuple(parts),supports
records=[];fits=[];max_active=0;minimum_active=48;max_grad=0
@torch.no_grad()
def evaluate(selector,label):
 global max_active,minimum_active
 pred={}
 for i,st in dev.items():
  pred[i],supp=selector(st);active=torch.stack([s.sum(1) for s in supp]).sum(0);max_active=max(max_active,float(active.max()));minimum_active=min(minimum_active,float(active.min()));assert all(torch.all((s==0)|(s==1)) for s in supp)
 for p in dp:
  for i,d in [p['indices'],p['indices'][::-1]]:
   y=suffix(pred[i]);base=native_dev[i];don=native_dev[d];yp=suffix(tuple(dev[i][k]+pred[d][k]-pred[i][k] for k in range(3)));groups=dd['examples'][i]['groups'];variant=dd['examples'][i]['variant'];records.append(dict(kind=selector.kind,objective=label,family=p['family'],row=i,whole_KL=float(KL(base.double(),y.double())),paired_KL=float(KL(don.double(),yp.double())),paired_denominator=float(KL(don.double(),base.double())),top1=int(y.argmax()) in groups[variant]))
for kind in kinds:
 evaluate(Selector(kind),'untrained')
 for objective in ['activation','finite_causal']:
  selector=Selector(kind);opt=torch.optim.Adam(selector.parameters(),lr=.03);history=[]
  for step,pid in enumerate(schedule):
   p=next(p for p in tp if p['id']==pid);i,d=p['indices'];ri,_=selector(train[i]);rd,_=selector(train[d])
   if objective=='activation':
    reconstruction=torch.stack([((rr[k]-train[ii][k])/scale[k]).square().mean() for ii,rr in [(i,ri),(d,rd)] for k in range(3)]).mean();paired=torch.stack([((rd[k]-ri[k]-(train[d][k]-train[i][k]))/scale[k]).square().mean()/(((train[d][k]-train[i][k])/scale[k]).square().mean()+1e-8) for k in range(3)]).mean();loss=reconstruction+.1*paired
   else:
    yi=suffix(ri);yd=suffix(rd);pi=suffix(tuple(train[i][k]+rd[k]-ri[k] for k in range(3)));pd=suffix(tuple(train[d][k]+ri[k]-rd[k] for k in range(3)));counts['training_student']+=4;reconstruction=(KL(native_train[i],yi)+KL(native_train[d],yd))/2;paired=(KL(native_train[d],pi)+KL(native_train[i],pd))/2;loss=reconstruction+paired
   loss=loss+1e-4*selector.importance.square().mean();opt.zero_grad();loss.backward();assert torch.isfinite(loss) and torch.isfinite(selector.importance.grad).all();max_grad=max(max_grad,float(selector.importance.grad.abs().max()));opt.step()
   with torch.no_grad():selector.importance.clamp_(-2,2)
   if step%20==0:history.append(dict(step=step,family=p['family'],whole=float(reconstruction.detach()),paired=float(paired.detach()),loss=float(loss.detach())));print('STEP',kind,objective,step,float(loss.detach()),flush=True)
  torch.save(dict(state=selector.state_dict(),kind=kind,objective=objective,history=history),out/f'{kind}_{objective}.pt');evaluate(selector,objective);fits.append(dict(kind=kind,objective=objective,history=history,importance_abs_max=float(selector.importance.abs().max()),parameters=selector.importance.numel()));print('DONE',kind,objective,flush=True)
summary=[]
for kind in kinds:
 for objective in ['untrained','activation','finite_causal']:
  for f in families:
   rr=[r for r in records if (r['kind'],r['objective'],r['family'])==(kind,objective,f)];summary.append(dict(kind=kind,objective=objective,family=f,whole_KL=float(np.mean([r['whole_KL'] for r in rr])),paired_relative_KL=sum(r['paired_KL'] for r in rr)/sum(r['paired_denominator'] for r in rr),top1=float(np.mean([r['top1'] for r in rr]))))
audit={'source_hashes_unchanged':all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p)] for p in [trainfile,pcafile]),'fits':len(fits),'max_active':max_active,'min_active':minimum_active,'finite':bool(np.isfinite([r['whole_KL'] for r in records]).all()),'max_gradient_abs':max_grad,'suffix_sequences':counts['suffix'],'training_student_suffix_sequences':counts['training_student'],'whole_task_holdout_reads':0};assert max_active<=48 and audit['finite'] and audit['source_hashes_unchanged'];(A/f'{S}.json').write_text(json.dumps(dict(config=config,summary=summary,records=records,fits=fits,audits=audit),indent=2)+'\n');print(json.dumps(audit,indent=2))

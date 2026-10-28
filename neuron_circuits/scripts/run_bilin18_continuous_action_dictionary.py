"""H1047 continuous decoder learning with fixed direct-change support."""
from pathlib import Path
import sys,json,hashlib,ast,numpy as np,torch
from torch import nn
R=Path('/home/loganriggs/Coding/neuron_circuits');A=R/'artifacts/bilin18_tensor_decomposition';S='bilin18_continuous_action_dictionary';out=A/'continuous_action_dictionary';out.mkdir(exist_ok=True)
repo=Path('/home/loganriggs/Coding/reverse_engineer/tensor_language')
for p in [repo,repo/'basis_aligned/polynomial_causal',repo/'basis_aligned/bilinear_quotient/ops']:sys.path.insert(0,str(p))
import bilin18_observed_model_facade as facade
F=torch.nn.functional;torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');seed=10331
files=[A/'multitask_boundary_contract/states.pt',A/'multitask_boundary_contract/panel.json',A/'diverse_task_expansion_screen/states.pt',A/'diverse_task_expansion_screen/training_eligibility.json',A/'bilin18_diverse_task_expansion_screen.json',A/'multistream_sparse_baselines/development_states.pt',A/'multistream_sparse_baselines/development_panel.json',A/'selector_task_holdouts/states.pt',A/'selector_task_holdouts/panel.json',A/'bilin18_selector_task_holdouts.json',A/'reserved_task_format_screen/states.pt',A/'bilin18_reserved_task_format_screen.json',R/'scripts/run_bilin18_finite_supervised_selectors.py'];hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
states={};examples={};native_saved={};trainpairs=[];evalpairs=[]
def ingest(prefix,data,rows,target,panel=None):
 for p in rows:
  indices=[]
  for i in p['indices']:
   k=f'{prefix}:{i}';states[k]=data['states'][i];examples[k]=data['examples'][i]
   if 'native' in data:native_saved[k]=data['native'][i]
   indices.append(k)
  target.append(dict(id=prefix+':'+p['id'],family=p['family'],indices=indices,panel=panel))
ingest('old',torch.load(files[0],weights_only=False),json.loads(files[1].read_text()),trainpairs)
eligible=set(json.loads(files[3].read_text())['pair_ids']);rows=[r for r in json.loads(files[4].read_text())['records'] if r['id'] in eligible];ingest('added',torch.load(files[2],weights_only=False),rows,trainpairs)
ingest('development',torch.load(files[5],weights_only=False),json.loads(files[6].read_text()),evalpairs,'same_family')
gates={(r['family'],r['split']) for r in json.loads(files[9].read_text())['competence'] if r['pass_gate']};rows=[r for r in json.loads(files[8].read_text()) if (r['family'],r['split']) in gates];ingest('historical',torch.load(files[7],weights_only=False),rows,evalpairs,'historical_excluded')
rows=[r for r in json.loads(files[11].read_text())['records'] if r['family']=='weekday_successor__prose'];ingest('weekday',torch.load(files[10],weights_only=False),rows,evalpairs,'weekday_development')
assert len(trainpairs)==80 and len(evalpairs)==39
traintexts={examples[i]['text'] for p in trainpairs for i in p['indices']};evaltexts={examples[i]['text'] for p in evalpairs for i in p['indices']};assert not traintexts&evaltexts
# Canonicalize repeated training endpoints by exact text. Reject inconsistent captures.
canonical={};aliases={};duplicate_max=0.
for p in trainpairs:
 for i in p['indices']:
  text=examples[i]['text']
  if text in canonical:
   k=canonical[text];assert examples[k]['ids']==examples[i]['ids'];duplicate_max=max(duplicate_max,max(float((a-b).abs().max()) for a,b in zip(states[k],states[i])));aliases[i]=k
  else:canonical[text]=i;aliases[i]=i
assert duplicate_max<1e-5
for p in trainpairs:p['indices']=[aliases[i] for i in p['indices']]
seen=set();unique=[]
for p in trainpairs:
 key=(p['family'],tuple(p['indices']))
 if key not in seen:unique.append(p);seen.add(key)
removed=len(trainpairs)-len(unique);trainpairs=unique
cohorts={'expanded':trainpairs};schedules={}
for c,pp in cohorts.items():
 fam=sorted({p['family'] for p in pp});schedules[c]=[next(p for p in [r for r in pp if r['family']==fam[step%len(fam)]] if p==[r for r in pp if r['family']==fam[step%len(fam)]][(step//len(fam))%len([r for r in pp if r['family']==fam[step%len(fam)]])])['id'] for step in range(160)]
config=dict(experiment='H1047',seed=seed,steps=160,source_hashes=hashes,schedule=schedules['expanded'],training='Expanded80pairs/10families, allpairs covered; identical family-balanced schedule across architectures.',geometry='Frozen H1042 PCA encoder coefficients and fixed untrained energy-based supports. Learn3x1152x128decoder weights,442368parameters each, with unit-column normalization.',objective='Bidirectional finite paired-edit KL plus1e-4mean squared column-direction displacement from initial PCA. No whole-state reconstruction objective or gate surrogate.',optimizer='Adam0.001,160fixed pair steps, final checkpoint only; no evaluation-based stopping or hyperparameter search.',controls='Initial fixed-support direct encoders replay H1044 within float32 column-normalization tolerance; exact continuous directional finite differences before fitting; full80pair hard-forward loss before/after.',evaluation='Same opened39pairs/78endpoints, allwhole-family exclusions preserved; no fresh-task discovery claim.',limits='Learned action decoder with frozen encoder and random group memberships, not a full BSF or learned manifold. Max48active coordinates, shared actualunion may be smaller.')
(R/f'configs/{S}.json').write_text(json.dumps(config,indent=2)+'\n')
tree=ast.parse(files[-1].read_text());nodes=[n for n in tree.body if (isinstance(n,ast.ClassDef) and n.name=='Selector') or (isinstance(n,ast.FunctionDef) and n.name in ['suffix','KL'])];exec(compile(ast.Module(body=nodes,type_ignores=[]),'<H1033-definitions>','exec'),globals())
model,_=facade.load_bilin18(snapshot=Path('/home/loganriggs/.cache/huggingface/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240'),device='cuda',dtype=torch.float32,verify_weights_sha256=True)
for p in model.parameters():p.requires_grad_(False)
counts={'suffix':0};trainix=sorted({i for p in trainpairs for i in p['indices']});evalix=sorted({i for p in evalpairs for i in p['indices']});states={i:tuple(v.cuda() for v in states[i]) for i in set(trainix+evalix)}
with torch.no_grad():native_train={i:suffix(states[i]).detach() for i in trainix};native_eval={i:native_saved[i].cuda() for i in evalix}
geometry=A/'training_coverage_selector_comparison/expanded_pca.pt';gh=hashlib.sha256(geometry.read_bytes()).hexdigest();pca=torch.load(geometry,weights_only=False);B=[b.cuda() for b in pca['bases']];scale=pca['scale'].cuda()
OriginalSelector=Selector
class Selector(nn.Module):
 def __init__(self,kind):
  super().__init__();self.kind=kind;self.register_buffer('membership',OriginalSelector(kind).membership.clone());self.dictionary=nn.Parameter(torch.stack(B).clone())
 def decoder(self):return F.normalize(self.dictionary,dim=1)
def direct(selector,delta):
 parts=[];supports=[];decoder=selector.decoder()
 for k in range(3):
  c=(delta[k].reshape(-1,1152)/scale[k])@B[k];scores=c.square()
  if selector.kind=='scalar':support=torch.zeros_like(c).scatter_(1,scores.topk(16,1).indices,1)
  else:
   M=selector.membership[k];gscore=scores@M.T/(4 if selector.kind=='block' else 8);budget=4 if selector.kind=='block' else 2;g=torch.zeros_like(gscore).scatter_(1,gscore.topk(budget,1).indices,1);support=(g@M>0).to(c)
  parts.append(((c*support)@decoder[k].T*scale[k]).reshape_as(delta[k]));supports.append(support)
 return tuple(parts),supports
@torch.no_grad()
def training_audit(selector,checkpoint):
 rows=[]
 for p in trainpairs:
  i,d=p['indices'];delta=tuple(states[d][k]-states[i][k] for k in range(3));change,_=direct(selector,delta);yi=suffix(tuple(states[i][k]+change[k] for k in range(3)));yd=suffix(tuple(states[d][k]-change[k] for k in range(3)));value=float((KL(native_train[d].double(),yi.double())+KL(native_train[i].double(),yd.double()))/2);rows.append(dict(kind=selector.kind,checkpoint=checkpoint,family=p['family'],pair=p['id'],KL=value))
 return rows
def pair_loss(selector,p):
 i,d=p['indices'];delta=tuple(states[d][k]-states[i][k] for k in range(3));change,_=direct(selector,delta);yi=suffix(tuple(states[i][k]+change[k] for k in range(3)));yd=suffix(tuple(states[d][k]-change[k] for k in range(3)));return (KL(native_train[d],yi)+KL(native_train[i],yd))/2
records=[];samples=[];fits=[];maxactive=0;minactive=48;maxgrad=0;replay=0;antisym=0.;zeroerr=0.;old=json.loads((A/'bilin18_direct_change_encoding.json').read_text());training_records=[];gradient_checks=[];column_norm_error=0.;initial_decoder=F.normalize(torch.stack(B),dim=1)
@torch.no_grad()
def evaluate(selector,objective):
 global maxactive,minactive,replay,antisym,zeroerr
 first={}
 for p in evalpairs:first.setdefault(p['panel'],p['indices'][0])
 for p in evalpairs:
  for i,d in [p['indices'],p['indices'][::-1]]:
   delta=tuple(states[d][k]-states[i][k] for k in range(3));change,ss=direct(selector,delta);active=torch.stack([s.sum(1) for s in ss]).sum(0);maxactive=max(maxactive,float(active.max()));minactive=min(minactive,float(active.min()));assert all(torch.all((s==0)|(s==1)) for s in ss)
   y=suffix(tuple(states[i][k]+change[k] for k in range(3)));base=native_eval[i];don=native_eval[d];val=float(KL(don.double(),y.double()));den=float(KL(don.double(),base.double()));g=examples[i]['groups'];v=examples[d]['variant'];records.append(dict(kind=selector.kind,objective=objective,panel=p['panel'],family=p['family'],row=i,donor=d,paired_KL=val,paired_denominator=den,donor_top1=int(y.argmax()) in g[v]))
   if objective=='untrained':
    prior=next(r for r in old['records'] if r['arm']==f'direct/{selector.kind}/untrained' and r['row']==i);replay=max(replay,abs(val-prior['paired_KL']))
   if i==first[p['panel']]:
    samples.append(dict(kind=selector.kind,objective=objective,row=i,base=base.cpu(),donor=don.cpu(),paired=y.double().cpu()));rev,_=direct(selector,tuple(-x for x in delta));zero,_=direct(selector,tuple(torch.zeros_like(x) for x in delta));antisym=max(antisym,max(float((a+b).abs().max()) for a,b in zip(change,rev)));zeroerr=max(zeroerr,max(float(x.abs().max()) for x in zero))
for kind in ['scalar','block','shared']:
 selector=Selector(kind);evaluate(selector,'untrained');assert replay<1e-5;training_records.extend(training_audit(selector,'initial'));history=[]
 # True continuous derivative through normalized decoder; supports do not depend on these weights.
 p=trainpairs[0];base_loss=pair_loss(selector,p);base_loss.backward();grad=selector.dictionary.grad.detach().clone();direction=-grad/(grad.norm()+1e-12);expected=float((grad*direction).sum());w0=selector.dictionary.detach().clone()
 with torch.no_grad():
  for eps in [.001,.01]:
   selector.dictionary.copy_(w0+eps*direction);plus=pair_loss(selector,p);selector.dictionary.copy_(w0-eps*direction);minus=pair_loss(selector,p);fd=float((plus-minus)/(2*eps));gradient_checks.append(dict(kind=kind,epsilon=eps,autograd=expected,finite_difference=fd,relative_error=abs(fd-expected)/max(abs(expected),1e-8)))
  selector.dictionary.copy_(w0)
 assert min(r['relative_error'] for r in gradient_checks if r['kind']==kind)<.02
 selector.zero_grad();opt=torch.optim.Adam(selector.parameters(),lr=.001)
 for step,pid in enumerate(schedules['expanded']):
  p=next(p for p in trainpairs if p['id']==pid);i,d=p['indices'];delta=tuple(states[d][k]-states[i][k] for k in range(3));change,_=direct(selector,delta);yi=suffix(tuple(states[i][k]+change[k] for k in range(3)));yd=suffix(tuple(states[d][k]-change[k] for k in range(3)));paired=(KL(native_train[d],yi)+KL(native_train[i],yd))/2;loss=paired+1e-4*(selector.decoder()-initial_decoder).square().sum(1).mean();opt.zero_grad();loss.backward();assert torch.isfinite(loss) and torch.isfinite(selector.dictionary.grad).all();maxgrad=max(maxgrad,float(selector.dictionary.grad.abs().max()));opt.step()
  if step%20==0:history.append(dict(step=step,family=p['family'],paired=float(paired.detach())));print('STEP',kind,step,flush=True)
 torch.save(dict(state=selector.state_dict(),optimizer=opt.state_dict(),kind=kind,steps=160,history=history),out/f'{kind}.pt');evaluate(selector,'direct_causal');training_records.extend(training_audit(selector,'final'));column_norm_error=max(column_norm_error,float((selector.decoder().norm(dim=1)-1).abs().max()));fits.append(dict(kind=kind,history=history));print('DONE',kind,flush=True)
summary=[]
for key in sorted({(r['kind'],r['objective'],r['panel'],r['family']) for r in records}):
 rr=[r for r in records if (r['kind'],r['objective'],r['panel'],r['family'])==key];summary.append(dict(zip(['kind','objective','panel','family'],key))|dict(paired_relative_KL=sum(r['paired_KL'] for r in rr)/sum(r['paired_denominator'] for r in rr),donor_top1=float(np.mean([r['donor_top1'] for r in rr]))))
audit=dict(source_hashes_unchanged=all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p)] for p in files),decoder_parameters=442368,decoder_unit_norm_error=column_norm_error,encoder_geometry_sha256=gh,encoder_geometry_unchanged=gh==hashlib.sha256(geometry.read_bytes()).hexdigest(),fits=3,distinct_supervised_pairs=len(set(schedules['expanded'])),untrained_H1044_KL_replay_max=replay,max_active=maxactive,min_active=minactive,direct_antisymmetry_max=antisym,direct_zero_max=zeroerr,max_gradient=maxgrad,suffix_sequences=counts['suffix'],training_student_suffix_sequences=960,train_eval_prompt_overlap=0,finite=all(np.isfinite(r['paired_KL']) for r in records))
assert audit['source_hashes_unchanged'] and audit['encoder_geometry_unchanged'] and audit['finite'] and maxactive<=48 and antisym<1e-6 and zeroerr==0
torch.save(samples,out/'audit_logits.pt');(A/f'{S}.json').write_text(json.dumps(dict(config=config,summary=summary,records=records,fits=fits,training_records=training_records,gradient_checks=gradient_checks,audits=audit),indent=2)+'\n');print(json.dumps(audit,indent=2))

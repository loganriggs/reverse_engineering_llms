"""H1045 direct-change finite-causal supervision at fixed geometry."""
from pathlib import Path
import sys,json,hashlib,ast,numpy as np,torch
from torch import nn
R=Path('/home/loganriggs/Coding/neuron_circuits');A=R/'artifacts/bilin18_tensor_decomposition';S='bilin18_direct_change_supervision';out=A/'direct_change_supervision';out.mkdir(exist_ok=True)
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
config=dict(experiment='H1045',seed=seed,steps=160,source_hashes=hashes,schedule=schedules['expanded'],training='Expanded80pairs/10families;16visits/family; allpairs covered. No evaluation fitting.',geometry='Exact frozen H1042 expanded PCA128perport; fixed H1033 group partitions.384learned log-importance coefficients. Max48active total.',objective='Mean full-donor KL for recipient+direct encoded donor-minus-recipient and reverse edit. No whole-state reconstruction objective.1e-4importance regularization.',optimizer='Adam0.03 from zero, clamp[-2,2], hardforward softmax straight-through gradient;160fixedsteps, final checkpoint only.',controls='Untrained direct encoders independently reproduce H1044; endpoint-trained transferred weights retained as historical controls, not initializations.',evaluation='Same39pairs/78endpoints across opened same-family,historical-excluded,weekday panels; no fresh-task or manifold discovery claim.')
(R/f'configs/{S}.json').write_text(json.dumps(config,indent=2)+'\n')
tree=ast.parse(files[-1].read_text());nodes=[n for n in tree.body if (isinstance(n,ast.ClassDef) and n.name=='Selector') or (isinstance(n,ast.FunctionDef) and n.name in ['suffix','KL'])];exec(compile(ast.Module(body=nodes,type_ignores=[]),'<H1033-definitions>','exec'),globals())
model,_=facade.load_bilin18(snapshot=Path('/home/loganriggs/.cache/huggingface/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240'),device='cuda',dtype=torch.float32,verify_weights_sha256=True)
for p in model.parameters():p.requires_grad_(False)
counts={'suffix':0};trainix=sorted({i for p in trainpairs for i in p['indices']});evalix=sorted({i for p in evalpairs for i in p['indices']});states={i:tuple(v.cuda() for v in states[i]) for i in set(trainix+evalix)}
with torch.no_grad():native_train={i:suffix(states[i]).detach() for i in trainix};native_eval={i:native_saved[i].cuda() for i in evalix}
geometry=A/'training_coverage_selector_comparison/expanded_pca.pt';gh=hashlib.sha256(geometry.read_bytes()).hexdigest();pca=torch.load(geometry,weights_only=False);B=[b.cuda() for b in pca['bases']];scale=pca['scale'].cuda()
def direct(selector,delta):
 parts=[];supports=[]
 for k in range(3):
  c=(delta[k].reshape(-1,1152)/scale[k])@B[k];energy=c.square();scores=energy*selector.importance[k].clamp(-2,2).exp();scores=scores/(energy.mean(1,keepdim=True).detach()+1e-8)
  if selector.kind=='scalar':gscore=scores;budget=16
  else:gscore=scores@selector.membership[k].T/(4 if selector.kind=='block' else 8);budget=4 if selector.kind=='block' else 2
  hard=torch.zeros_like(gscore).scatter_(1,gscore.topk(budget,1).indices,1);soft=gscore.softmax(1)*budget;gates=hard+(soft-soft.detach())
  if selector.kind=='scalar':support=gates
  else:
   hs=(hard@selector.membership[k]>0).to(c);ss=gates@selector.membership[k];support=hs+(ss-ss.detach())
  parts.append(((c*support)@B[k].T*scale[k]).reshape_as(delta[k]));supports.append(support)
 return tuple(parts),supports
records=[];samples=[];fits=[];maxactive=0;minactive=48;maxgrad=0;replay=0;antisym=0.;zeroerr=0.;old=json.loads((A/'bilin18_direct_change_encoding.json').read_text())
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
 selector=Selector(kind);evaluate(selector,'untrained');assert replay<1e-8;opt=torch.optim.Adam(selector.parameters(),lr=.03);history=[]
 for step,pid in enumerate(schedules['expanded']):
  p=next(p for p in trainpairs if p['id']==pid);i,d=p['indices'];delta=tuple(states[d][k]-states[i][k] for k in range(3));change,_=direct(selector,delta);yi=suffix(tuple(states[i][k]+change[k] for k in range(3)));yd=suffix(tuple(states[d][k]-change[k] for k in range(3)));paired=(KL(native_train[d],yi)+KL(native_train[i],yd))/2;loss=paired+1e-4*selector.importance.square().mean();opt.zero_grad();loss.backward();assert torch.isfinite(loss) and torch.isfinite(selector.importance.grad).all();maxgrad=max(maxgrad,float(selector.importance.grad.abs().max()));opt.step()
  with torch.no_grad():selector.importance.clamp_(-2,2)
  if step%20==0:history.append(dict(step=step,family=p['family'],paired=float(paired.detach())));print('STEP',kind,step,flush=True)
 torch.save(dict(state=selector.state_dict(),optimizer=opt.state_dict(),kind=kind,steps=160,history=history),out/f'{kind}.pt');evaluate(selector,'direct_causal');fits.append(dict(kind=kind,history=history));print('DONE',kind,flush=True)
summary=[]
for key in sorted({(r['kind'],r['objective'],r['panel'],r['family']) for r in records}):
 rr=[r for r in records if (r['kind'],r['objective'],r['panel'],r['family'])==key];summary.append(dict(zip(['kind','objective','panel','family'],key))|dict(paired_relative_KL=sum(r['paired_KL'] for r in rr)/sum(r['paired_denominator'] for r in rr),donor_top1=float(np.mean([r['donor_top1'] for r in rr]))))
audit=dict(source_hashes_unchanged=all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p)] for p in files),geometry_sha256=gh,geometry_unchanged=gh==hashlib.sha256(geometry.read_bytes()).hexdigest(),fits=3,distinct_supervised_pairs=len(set(schedules['expanded'])),untrained_H1044_KL_replay_max=replay,max_active=maxactive,min_active=minactive,direct_antisymmetry_max=antisym,direct_zero_max=zeroerr,max_gradient=maxgrad,suffix_sequences=counts['suffix'],training_student_suffix_sequences=960,train_eval_prompt_overlap=0,finite=all(np.isfinite(r['paired_KL']) for r in records))
assert audit['source_hashes_unchanged'] and audit['geometry_unchanged'] and audit['finite'] and maxactive<=48 and antisym<1e-6 and zeroerr==0
torch.save(samples,out/'audit_logits.pt');(A/f'{S}.json').write_text(json.dumps(dict(config=config,summary=summary,records=records,fits=fits,audits=audit),indent=2)+'\n');print(json.dumps(audit,indent=2))

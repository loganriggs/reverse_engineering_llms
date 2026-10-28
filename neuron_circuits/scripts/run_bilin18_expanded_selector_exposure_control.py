"""H1043 expanded160 exposure control with verified80step optimizer replay."""
from pathlib import Path
import sys,json,hashlib,ast,numpy as np,torch
from torch import nn
R=Path('/home/loganriggs/Coding/neuron_circuits');A=R/'artifacts/bilin18_tensor_decomposition';S='bilin18_expanded_selector_exposure_control';out=A/'expanded_selector_exposure_control';out.mkdir(exist_ok=True)
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
config=dict(experiment='H1043',seed=seed,steps=160,arms='expanded160 scalar/block/shared, compared with saved narrow80 and expanded80. First80replayed from initialization to reconstruct Adam; checkpoint equality required.',geometry='Reuse exact H1042 expanded PCA128 perport and training-only center/scale; no geometry refit.',optimization='Exact H1033 Selector class, Adam0.03,160fixed pair steps, family round-robin,1e-4importance penalty, clamp[-2,2].16visits/family matches narrow80;2xoptimization compute. No evaluation-dependent stopping.',scope='Fixed random partitions of PCA coordinates;384learned importance values, not learned semantic memberships or a full learned featurizer. Single seed pilot.',capacity='128dictionary atoms perstream,16active max perstream,48total; shared unions may use fewer.',objective='Mean native-to-reconstruction KL plus mean full-donor-to-decoded-difference-patch KL, both directions. Native suffix only for supervision/evaluation.',evaluations='Opened same-family19pairs, historical excluded16pairs, adaptively screened weekday4pairs. No fresh-task generalization claim.',source_hashes=hashes,schedules=schedules,duplicate_endpoint_aliases=len(aliases)-len(canonical),duplicate_pairs_removed=removed,train_eval_prompt_overlap=0)
(R/f'configs/{S}.json').write_text(json.dumps(config,indent=2)+'\n');(out/'panel.json').write_text(json.dumps(dict(train=trainpairs,evaluation=evalpairs),indent=2)+'\n')
# Load definitions only; do not execute the previous training driver.
tree=ast.parse(files[-1].read_text());nodes=[n for n in tree.body if (isinstance(n,ast.ClassDef) and n.name=='Selector') or (isinstance(n,ast.FunctionDef) and n.name in ['suffix','KL'])];exec(compile(ast.Module(body=nodes,type_ignores=[]),'<H1033-definitions>','exec'),globals())
model,_=facade.load_bilin18(snapshot=Path('/home/loganriggs/.cache/huggingface/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240'),device='cuda',dtype=torch.float32,verify_weights_sha256=True)
for p in model.parameters():p.requires_grad_(False)
counts={'suffix':0,'training_student':0};trainix=sorted({i for p in trainpairs for i in p['indices']});evalix=sorted({i for p in evalpairs for i in p['indices']});states={i:tuple(v.cuda() for v in states[i]) for i in set(trainix+evalix)}
with torch.no_grad():native_train={i:suffix(states[i]).detach() for i in trainix};native_eval={i:native_saved[i].cuda() for i in evalix}
records=[];samples=[];fitlogs=[];maxactive=0;minactive=48;maxgrad=0;orth=0;replay_errors={};frozenroot=A/'training_coverage_selector_comparison';frozenfiles=[frozenroot/'expanded_pca.pt']+[frozenroot/f'expanded_{k}.pt' for k in ['scalar','block','shared']];frozenhashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in frozenfiles}
@torch.no_grad()
def evaluate(selector,cohort,objective):
 global maxactive,minactive
 pred={}
 for i in evalix:
  pred[i],supp=selector(states[i]);a=torch.stack([s.sum(1) for s in supp]).sum(0);maxactive=max(maxactive,float(a.max()));minactive=min(minactive,float(a.min()));assert all(torch.all((s==0)|(s==1)) for s in supp)
 first={}
 for p in evalpairs:first.setdefault(p['panel'],p['indices'][0])
 for p in evalpairs:
  for i,d in [p['indices'],p['indices'][::-1]]:
   y=suffix(pred[i]);yp=suffix(tuple(states[i][k]+pred[d][k]-pred[i][k] for k in range(3)));base=native_eval[i];don=native_eval[d];g=examples[i]['groups'];v=examples[i]['variant'];records.append(dict(cohort=cohort,kind=selector.kind,objective=objective,panel=p['panel'],family=p['family'],row=i,donor=d,whole_KL=float(KL(base.double(),y.double())),paired_KL=float(KL(don.double(),yp.double())),paired_denominator=float(KL(don.double(),base.double())),top1=int(y.argmax()) in g[v]))
   if i==first[p['panel']]:samples.append(dict(cohort=cohort,kind=selector.kind,objective=objective,row=i,base=base.cpu(),donor=don.cpu(),whole=y.double().cpu(),paired=yp.double().cpu()))
for cohort,pp in cohorts.items():
 savedpca=torch.load(frozenroot/'expanded_pca.pt',weights_only=False);B=[b.cuda() for b in savedpca['bases']];mu=savedpca['center'].cuda();scale=savedpca['scale'].cuda()
 for kind in ['scalar','block','shared']:
  selector=Selector(kind);opt=torch.optim.Adam(selector.parameters(),lr=.03);history=[]
  for step,pid in enumerate(schedules[cohort]):
   p=next(r for r in pp if r['id']==pid);i,d=p['indices'];ri,_=selector(states[i]);rd,_=selector(states[d]);yi=suffix(ri);yd=suffix(rd);pi=suffix(tuple(states[i][k]+rd[k]-ri[k] for k in range(3)));pd=suffix(tuple(states[d][k]+ri[k]-rd[k] for k in range(3)));counts['training_student']+=4;rec=(KL(native_train[i],yi)+KL(native_train[d],yd))/2;pair=(KL(native_train[d],pi)+KL(native_train[i],pd))/2;loss=rec+pair+1e-4*selector.importance.square().mean();opt.zero_grad();loss.backward();assert torch.isfinite(loss) and torch.isfinite(selector.importance.grad).all();maxgrad=max(maxgrad,float(selector.importance.grad.abs().max()));opt.step()
   with torch.no_grad():selector.importance.clamp_(-2,2)
   if step==79:
    prior=torch.load(frozenroot/f'expanded_{kind}.pt',weights_only=False)['state'];replay_errors[kind]=max(float((v-prior[k].cuda()).abs().max()) for k,v in selector.state_dict().items());assert replay_errors[kind]<1e-7, replay_errors
    torch.save(dict(state=selector.state_dict(),optimizer=opt.state_dict(),step=80),out/f'{kind}_step80_replayed.pt')
   if step%20==0:history.append(dict(step=step,family=p['family'],whole=float(rec.detach()),paired=float(pair.detach())));print('STEP',cohort,kind,step,flush=True)
  torch.save(dict(state=selector.state_dict(),optimizer=opt.state_dict(),step=160,kind=kind,cohort=cohort,history=history),out/f'{cohort}_{kind}.pt');evaluate(selector,cohort,'finite_causal');fitlogs.append(dict(cohort=cohort,kind=kind,history=history));print('DONE',cohort,kind,flush=True)
summary=[]
for key in sorted({(r['cohort'],r['kind'],r['objective'],r['panel'],r['family']) for r in records}):
 rr=[r for r in records if (r['cohort'],r['kind'],r['objective'],r['panel'],r['family'])==key];summary.append(dict(zip(['cohort','kind','objective','panel','family'],key))|dict(whole_KL=float(np.mean([r['whole_KL'] for r in rr])),paired_relative_KL=sum(r['paired_KL'] for r in rr)/sum(r['paired_denominator'] for r in rr),top1=float(np.mean([r['top1'] for r in rr]))))
audit=dict(step80_weight_replay_errors=replay_errors,frozen_checkpoint_hashes=frozenhashes,frozen_checkpoints_unchanged=all(hashlib.sha256(p.read_bytes()).hexdigest()==frozenhashes[str(p)] for p in frozenfiles),distinct_supervised_pairs=len(set(schedules['expanded'])),source_hashes_unchanged=all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p)] for p in files),fits=len(fitlogs),max_active=maxactive,min_active=minactive,max_gradient=maxgrad,geometry_refitted=False,duplicate_capture_max=duplicate_max,train_eval_prompt_overlap=0,training_unique_prompts=len(trainix),evaluation_endpoints=len(evalix),suffix_sequences=counts['suffix'],training_student_suffix_sequences=counts['training_student'],finite=all(np.isfinite(r['whole_KL']) and np.isfinite(r['paired_KL']) for r in records))
assert audit['source_hashes_unchanged'] and audit['finite'] and maxactive<=48 and audit['frozen_checkpoints_unchanged'] and audit['distinct_supervised_pairs']==80
torch.save(samples,out/'audit_logits.pt');(A/f'{S}.json').write_text(json.dumps(dict(config=config,summary=summary,records=records,fits=fitlogs,audits=audit),indent=2)+'\n');print(json.dumps(audit,indent=2))

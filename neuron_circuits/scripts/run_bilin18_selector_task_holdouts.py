"""H1034 frozen-selector transfer to task-training-excluded historical families."""
from pathlib import Path
import sys,json,hashlib,ast,numpy as np,torch
from torch import nn
R=Path('/home/loganriggs/Coding/neuron_circuits');A=R/'artifacts/bilin18_tensor_decomposition';S='bilin18_selector_task_holdouts';out=A/'selector_task_holdouts';out.mkdir(exist_ok=True)
repo=Path('/home/loganriggs/Coding/reverse_engineer/tensor_language')
for p in [repo,repo/'basis_aligned/polynomial_causal',repo/'basis_aligned/bilinear_quotient/ops']:sys.path.insert(0,str(p))
import run_subject_number_sparse_graph_token_extraction_v1 as runner
import bilin18_observed_model_facade as facade
torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');F=torch.nn.functional
bank=json.loads((A/'bilin18_expanded_task_bank.json').read_text());data=torch.load(A/'bilin18_expanded_task_bank_captures.pt',weights_only=False);families=['sequence_copy','spatial_inverse'];allpairs=[r for r in bank['records'] if r['family'] in families];pairs=[r for r in allpairs if r['equal_length']];ix=sorted({i for p in pairs for i in p['indices']});assert len(pairs)==32 and len(ix)==64
train=torch.load(A/'multitask_boundary_contract/states.pt',weights_only=False);traintexts={e['text'] for e in train['examples'].values()};assert not any(data['examples'][i]['text'] in traintexts for i in ix);assert all(p['whole_task_holdout'] for p in pairs)
pcafile=A/'multistream_span_diagnosis/pca.pt';selectorfile=R/'scripts/run_bilin18_finite_supervised_selectors.py';files=[pcafile,selectorfile]+[A/f'finite_supervised_selectors/{k}_{o}.pt' for k in ['scalar','block','shared'] for o in ['activation','finite_causal']];hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files};config={'experiment':'H1034','families':families,'selection':'All32pairs/64endpoints across discovery,lexical,structural,joint historical splits; no per-outcome filtering. These entire families were excluded from H1031-H1033training and PCA. Historically evaluated earlier; not untouched discovery.','candidates':'All9 H1033 selectors (untrained,activation,finite_causal x scalar,block,shared), plus frozen PCA16 and PCA128 perstream. PCA128 is384coordinate reference, not48budget matched.','competence':'Per family/split fourdistinctpairs,>=75%both-endpoint accuracy and>=50%intended answer top1. Failed cohorts retained diagnostic.','training':'None. No refitting, model selection, rank/threshold/temperature adjustment.','source_hashes':hashes};(R/f'configs/{S}.json').write_text(json.dumps(config,indent=2)+'\n');(out/'panel.json').write_text(json.dumps(pairs,indent=2)+'\n')
pca=torch.load(pcafile,weights_only=False);B=[b.float().cuda() for b in pca['bases']];mu=pca['center'].float().cuda();scale=pca['scale'].float().cuda();seed=10331
# Load only the exact frozen class definition, never the source script's training driver.
tree=ast.parse(selectorfile.read_text());node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Selector');exec(compile(ast.Module(body=[node],type_ignores=[]),str(selectorfile),'exec'),globals())
model,_=facade.load_bilin18(snapshot=Path('/home/loganriggs/.cache/huggingface/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240'),device='cuda',dtype=torch.float32,verify_weights_sha256=True)
@torch.no_grad()
def capture(i):
 initial=F.rms_norm(model.transformer.wte(torch.tensor([data['examples'][i]['ids']],device='cuda')),(1152,)).float();raw,x0,bus,_,_=runner.component._capture(model,initial,torch,F);return raw,x0,bus
@torch.no_grad()
def suffix(st):
 x,x0,bus=st
 for l in range(11,18):
  b=model.transformer.h[l]
  if l>11:x=b.lambdas[0]*x+b.lambdas[1]*x0
  at,_=b.attn(F.rms_norm(x,(1152,)),bus);x=x+at;x=x+b.mlp(F.rms_norm(x,(1152,)))
 return (30*torch.tanh(model.lm_head(F.rms_norm(x,(1152,))[:,-1])/30))[0].double()
def KL(a,b):lp=F.log_softmax(a,0);return float((lp.exp()*(lp-F.log_softmax(b,0))).sum())
def margin(y,g):return float(torch.logsumexp(y[g[0]],0)-torch.logsumexp(y[g[1]],0))
states={i:capture(i) for i in ix};native={i:suffix(states[i]) for i in ix};competence=[]
for f in families:
 for split in ['discovery','lexical','structural','joint']:
  pp=[p for p in pairs if p['family']==f and p['split']==split];both=[];top=[]
  for p in pp:
   both.append(all(margin(native[i],data['examples'][i]['groups'])*(1 if end==0 else -1)>0 for end,i in enumerate(p['indices'])))
   top.extend(int(native[i].argmax()) in data['examples'][i]['groups'][end] for end,i in enumerate(p['indices']))
  competence.append(dict(family=f,split=split,both_endpoint_accuracy=float(np.mean(both)),top1=float(np.mean(top)),pass_gate=bool(len(pp)>=4 and np.mean(both)>=.75 and np.mean(top)>=.5)))
gates={(r['family'],r['split']):r['pass_gate'] for r in competence};records=[];samples=[];maxactive=0
with torch.no_grad():
 for name in [f'{k}/{o}' for k in ['scalar','block','shared'] for o in ['untrained','activation','finite_causal']]+['PCA16','PCA128']:
  if not name.startswith('PCA'):
   kind,objective=name.split('/');m=Selector(kind)
   if objective!='untrained':m.load_state_dict(torch.load(A/f'finite_supervised_selectors/{kind}_{objective}.pt',weights_only=False)['state'])
  pred={}
  for i,st in states.items():
   if name.startswith('PCA'):
    rank=int(name[3:]);parts=[]
    for k in range(3):z=(st[k].reshape(-1,1152)-mu[k])/scale[k];parts.append(((z@B[k][:,:rank])@B[k][:,:rank].T*scale[k]+mu[k]).reshape_as(st[k]))
    pred[i]=tuple(parts)
   else:
    pred[i],supp=m(st);assert all(torch.all((s==0)|(s==1)) for s in supp);maxactive=max(maxactive,float(torch.stack([s.sum(1) for s in supp]).sum(0).max()))
  for p in pairs:
   for i,d in [p['indices'],p['indices'][::-1]]:
    y=suffix(pred[i]);base=native[i];don=native[d];yp=suffix(tuple(states[i][k]+pred[d][k]-pred[i][k] for k in range(3)));g=data['examples'][i]['groups'];end=p['indices'].index(i);records.append(dict(arm=name,family=p['family'],split=p['split'],row=i,donor=d,pass_native_gate=gates[p['family'],p['split']],whole_KL=KL(base,y),paired_KL=KL(don,yp),paired_denominator=KL(don,base),natural_delta=margin(don,g)-margin(base,g),predicted_delta=margin(yp,g)-margin(base,g),correct_top1=int(y.argmax()) in g[end]))
    if i==pairs[0]['indices'][0]:samples.append(dict(arm=name,row=i,base=base.cpu(),donor=don.cpu(),whole=y.cpu(),paired=yp.cpu()))
  print('EVALUATED',name,flush=True)
summary=[]
for name in dict.fromkeys(r['arm'] for r in records):
 for f in families:
  for split in ['discovery','lexical','structural','joint']:
   rr=[r for r in records if (r['arm'],r['family'],r['split'])==(name,f,split)];nd=np.array([r['natural_delta'] for r in rr]);pd=np.array([r['predicted_delta'] for r in rr]);summary.append(dict(arm=name,family=f,split=split,pass_native_gate=gates[f,split],whole_KL=float(np.mean([r['whole_KL'] for r in rr])),paired_relative_KL=sum(r['paired_KL'] for r in rr)/sum(r['paired_denominator'] for r in rr),paired_margin_relative_l2=float(np.linalg.norm(pd-nd)/np.linalg.norm(nd)),top1=float(np.mean([r['correct_top1'] for r in rr]))))
audit={'source_hashes_unchanged':all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p)] for p in files),'train_eval_prompt_overlap':0,'fit_rows':0,'prefix_sequences':64,'suffix_sequences':1472,'max_selector_active':maxactive,'finite':bool(np.isfinite([r['whole_KL'] for r in records]).all())};assert audit['source_hashes_unchanged'] and maxactive<=48 and audit['finite'];torch.save(dict(states={i:tuple(v.cpu() for v in s) for i,s in states.items()},native={i:v.cpu() for i,v in native.items()},examples={i:data['examples'][i] for i in ix}),out/'states.pt');torch.save(samples,out/'audit_logits.pt');(A/f'{S}.json').write_text(json.dumps(dict(config=config,competence=competence,summary=summary,records=records,audits=audit),indent=2)+'\n');print(json.dumps(dict(competence=competence,audits=audit),indent=2))

"""H1049 frozen factual crossfit transfer, shared-training versus excluded countries."""
from pathlib import Path
import sys,json,hashlib,ast,re,numpy as np,torch
from torch import nn
from transformers import AutoTokenizer
R=Path('/home/loganriggs/Coding/neuron_circuits');A=R/'artifacts/bilin18_tensor_decomposition';S='bilin18_factual_cross_country_transfer';out=A/'factual_cross_country_transfer';out.mkdir(exist_ok=True)
repo=Path('/home/loganriggs/Coding/reverse_engineer/tensor_language')
for p in [repo,repo/'basis_aligned/polynomial_causal',repo/'basis_aligned/bilinear_quotient/ops']:sys.path.insert(0,str(p))
import bilin18_observed_model_facade as facade
import run_subject_number_sparse_graph_token_extraction_v1 as runner
F=torch.nn.functional;torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');seed=10331
tok=AutoTokenizer.from_pretrained('/home/loganriggs/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e',local_files_only=True)
seen=[('France','Paris','French'),('Germany','Berlin','German'),('Italy','Rome','Italian'),('Japan','Tokyo','Japanese'),('Spain','Madrid','Spanish'),('China','Beijing','Chinese'),('Russia','Moscow','Russian'),('France','Paris','French')]
unseen=[('Poland','Warsaw','Polish'),('Turkey','Ankara','Turkish'),('Finland','Helsinki','Finnish'),('Sweden','Stockholm','Swedish'),('Norway','Oslo','Norwegian'),('Portugal','Lisbon','Portuguese'),('Denmark','Copenhagen','Danish'),('Hungary','Budapest','Hungarian')]
folds=['capital_recall','country_language'];templates=['The capital city of {} is','The main language spoken in {} is'];panel=[];examples={}
for f,template,col in zip(folds,templates,[1,2]):
 for cohort,rows in [('seen_countries',seen),('excluded_countries',unseen)]:
  for n in range(4):
   pair=rows[2*n:2*n+2];ii=[]
   for variant,row in enumerate(pair):
    text=template.format(row[0]);ids=tok.encode(text,add_special_tokens=False);gg=[]
    for answer in [r[col] for r in pair]:
     g=[]
     for form in dict.fromkeys([answer,answer.lower(),answer.upper()]):
      enc=tok.encode(text+' '+form,add_special_tokens=False)
      if len(enc)==len(ids)+1 and enc[:len(ids)]==ids:g.append(enc[-1])
     gg.append(sorted(set(g)))
    assert all(gg) and not set(gg[0])&set(gg[1]);i=len(examples);ii.append(i);examples[i]=dict(text=text,ids=ids,groups=gg,variant=variant,country=row[0])
   assert len(examples[ii[0]]['ids'])==len(examples[ii[1]]['ids']);panel.append(dict(id=f'{f}/{cohort}/{n}',family=f,cohort=cohort,indices=ii,countries=[r[0] for r in pair]))
# Verify country exclusion against all actual prompt texts used to fit each fold.
trainpanel=json.loads((A/'training_coverage_selector_comparison/panel.json').read_text())['train'];datasets={'old':torch.load(A/'multitask_boundary_contract/states.pt',weights_only=False),'added':torch.load(A/'diverse_task_expansion_screen/states.pt',weights_only=False)};lexical_audit={}
for f in folds:
 texts=[]
 for p in trainpanel:
  if p['family']==f:continue
  for i in p['indices']:
   prefix,idx=i.split(':');texts.append(datasets[prefix]['examples'][int(idx)]['text'])
 present={country:any(re.search(r'\b'+re.escape(country)+r'\b',t) for t in texts) for country,_,_ in seen+unseen};assert all(present[r[0]] for r in seen) and not any(present[r[0]] for r in unseen);lexical_audit[f]=present
root=A/'action_dictionary_family_crossfit';files=[root/f/name for f in folds for name in ['geometry.pt','scalar.pt','block.pt','shared.pt']];hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
config=dict(experiment='H1049',source_hashes=hashes,fit_rows=0,selection='Fixed4pairs per cohort/task; seven shared-training country identities (France repeated), eight excluded countries. No model-output filtering. Same wording for seen/excluded.',competence='Report both-endpoint accuracy and unrestricted top1 separately; fixed >=75%both/50%top1 for fourpair cohort. No failed cohort deleted.',limits='Countries historically evaluated elsewhere; exclusion is from this fold fitting, not model pretraining or global research. Seven vs eight unique identities and repeated France reduce independence. Related task remains in training. Paired lexical contrast, not content-independent operation proof.',lexical_exclusion_audit=lexical_audit)
(R/f'configs/{S}.json').write_text(json.dumps(config,indent=2)+'\n');(out/'panel.json').write_text(json.dumps(dict(pairs=panel,examples=examples),indent=2)+'\n');ph=hashlib.sha256((out/'panel.json').read_bytes()).hexdigest()
for file,names in [(R/'scripts/run_bilin18_finite_supervised_selectors.py',['Selector','suffix','KL']),(R/'scripts/run_bilin18_continuous_action_dictionary.py',['Selector','direct'])]:
 if file.name=='run_bilin18_continuous_action_dictionary.py':OriginalSelector=Selector
 tree=ast.parse(file.read_text());nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names];exec(compile(ast.Module(body=nodes,type_ignores=[]),str(file),'exec'),globals())
counts={'suffix':0};model,_=facade.load_bilin18(snapshot=Path('/home/loganriggs/.cache/huggingface/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240'),device='cuda',dtype=torch.float32,verify_weights_sha256=True)
states={};native={};records=[];samples=[]
with torch.no_grad():
 for i,e in examples.items():
  initial=F.rms_norm(model.transformer.wte(torch.tensor([e['ids']],device='cuda')),(1152,)).float();raw,x0,bus,_,_=runner.component._capture(model,initial,torch,F);states[i]=(raw,x0,bus);native[i]=suffix(states[i]).double()
 for f in folds:
  geo=torch.load(root/f/'geometry.pt',weights_only=False);B=[v.cuda() for v in geo['bases']];scale=geo['scale'].cuda()
  for kind,checkpoint in [(k,c) for k in ['scalar','block','shared'] for c in ['initial','final']]+[('dense','initial')]:
   if kind!='dense':
    selector=Selector(kind)
    if checkpoint=='final':selector.load_state_dict(torch.load(root/f/f'{kind}.pt',weights_only=False)['state'])
   for p in [p for p in panel if p['family']==f]:
    for i,d in [p['indices'],p['indices'][::-1]]:
     delta=tuple(states[d][k]-states[i][k] for k in range(3))
     if kind=='dense':change=tuple((((delta[k].reshape(-1,1152)/scale[k])@B[k])@B[k].T*scale[k]).reshape_as(delta[k]) for k in range(3))
     else:change,_=direct(selector,delta)
     y=suffix(tuple(states[i][k]+change[k] for k in range(3))).double();base=native[i];don=native[d];g=examples[i]['groups'];records.append(dict(family=f,cohort=p['cohort'],kind=kind,checkpoint=checkpoint,row=i,donor=d,paired_KL=float(KL(don,y)),paired_denominator=float(KL(don,base)),donor_top1=int(y.argmax()) in g[examples[d]['variant']]))
     if i==p['indices'][0] and p['id'].endswith('/0'):samples.append(dict(family=f,cohort=p['cohort'],kind=kind,checkpoint=checkpoint,row=i,base=base.cpu(),donor=don.cpu(),paired=y.cpu()))
competence=[];summary=[]
for f in folds:
 for cohort in ['seen_countries','excluded_countries']:
  pp=[p for p in panel if p['family']==f and p['cohort']==cohort];both=[];top=[]
  for p in pp:
   correct=[]
   for variant,i in enumerate(p['indices']):
    g=examples[i]['groups'];y=native[i];m=float(y[g[0]].logsumexp(0)-y[g[1]].logsumexp(0));correct.append(m*(1 if variant==0 else -1)>0);top.append(int(y.argmax()) in g[variant])
   both.append(all(correct))
  competence.append(dict(family=f,cohort=cohort,both_endpoint_accuracy=float(np.mean(both)),top1=float(np.mean(top)),pass_gate=np.mean(both)>=.75 and np.mean(top)>=.5))
for key in sorted({(r['family'],r['cohort'],r['kind'],r['checkpoint']) for r in records}):
 rr=[r for r in records if (r['family'],r['cohort'],r['kind'],r['checkpoint'])==key];summary.append(dict(zip(['family','cohort','kind','checkpoint'],key))|dict(paired_relative_KL=sum(r['paired_KL'] for r in rr)/sum(r['paired_denominator'] for r in rr),donor_top1=float(np.mean([r['donor_top1'] for r in rr]))))
audit=dict(source_hashes_unchanged=all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p)] for p in files),panel_hash_unchanged=ph==hashlib.sha256((out/'panel.json').read_bytes()).hexdigest(),excluded_country_mentions_in_training=0,prefix_sequences=32,suffix_sequences=counts['suffix'],fit_rows=0,finite=all(np.isfinite(r['paired_KL']) for r in records));assert audit['source_hashes_unchanged'] and audit['panel_hash_unchanged'] and audit['finite']
# Convert numpy bool for JSON; retain every cohort including failed native competence.
for c in competence:c['pass_gate']=bool(c['pass_gate'])
torch.save(dict(states={i:tuple(v.cpu() for v in st) for i,st in states.items()},native={i:y.cpu() for i,y in native.items()},examples=examples),out/'states.pt');torch.save(samples,out/'audit_logits.pt');(A/f'{S}.json').write_text(json.dumps(dict(config=config,competence=competence,summary=summary,records=records,audits=audit),indent=2)+'\n');print(json.dumps(dict(competence=competence,audits=audit),indent=2))

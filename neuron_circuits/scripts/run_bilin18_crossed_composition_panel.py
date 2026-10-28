"""H1053 frozen four-task content x context checkerboard, native and causal controls."""
from pathlib import Path
import sys,json,hashlib,ast,importlib.util,numpy as np,torch
R=Path('/home/loganriggs/Coding/neuron_circuits');A=R/'artifacts/bilin18_tensor_decomposition';S='bilin18_crossed_composition_panel';out=A/'crossed_composition_panel';out.mkdir(exist_ok=True)
repo=Path('/home/loganriggs/Coding/reverse_engineer/tensor_language')
for p in [repo,repo/'basis_aligned/polynomial_causal',repo/'basis_aligned/bilinear_quotient/ops']:sys.path.insert(0,str(p))
import bilin18_observed_model_facade as facade
import run_subject_number_sparse_graph_token_extraction_v1 as runner
from transformers import AutoTokenizer
F=torch.nn.functional;torch.set_num_threads(6);torch.set_float32_matmul_precision('highest')
tok=AutoTokenizer.from_pretrained('/home/loganriggs/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e',local_files_only=True)
specs={
 'agreement':dict(items=[('boy','boys'),('girl','girls'),('dancer','dancers'),('singer','singers')],templates=['The {x} outside the museum','The {x} behind the station','The {x} beside the fountain','The {x} near the garden'],answers=[['is','are']]*4,variable='subject number; content factor is noun stem; context factor is location phrase'),
 'pronoun_gender':dict(items=[('man','woman'),('boy','girl'),('father','mother'),('king','queen')],templates=['The {x} outside the museum said that','The {x} behind the station said that','The {x} beside the fountain said that','The {x} near the garden said that'],answers=[['he','she']]*4,variable='referent gender; content factor is gendered referent pair; context factor is location phrase'),
 'sentiment':dict(items=[('excellent','terrible'),('good','bad'),('wonderful','awful'),('great','poor')],templates=['Customer note: The documentary was {x}. Overall sentiment:','Customer note: The hotel was {x}. Overall sentiment:','Customer note: The restaurant was {x}. Overall sentiment:','Customer note: The service was {x}. Overall sentiment:'],answers=[['positive','negative']]*4,variable='evaluative adjective polarity; content factor is adjective pair; context factor is reviewed object'),
 'antonym':dict(items=[('hot','cold'),('big','small'),('up','down'),('good','bad')],templates=['Vocabulary quiz: the opposite of {x} is','Word puzzle: the opposite of {x} is','In this lesson, the opposite of {x} is','In this exercise, the opposite of {x} is'],answers=[['cold','hot'],['small','big'],['down','up'],['bad','good']],variable='queried lexical identity; content factor is opposite pair; context factor is framing prefix')}
manifest=[];examples={};pairs=[];excluded=[]
for family,spec in specs.items():
 for c,template in enumerate(spec['templates']):
  for source,item in enumerate(spec['items']):
   row=dict(id=f'{family}/context{c}/content{source}',family=family,context=c,content=source,split='train' if (c+source)%2==0 else 'combination',prompts=[template.format(x=x) for x in item],answers=spec['answers'][source],variable=spec['variable']);manifest.append(row);ee=[]
   if family=='agreement':strings=[[' is',' was',' has',' does'],[' are',' were',' have',' do']]
   elif family=='sentiment':strings=[[' positive',' good',' Good',' great',' Great',' excellent',' favorable'],[' negative',' bad',' Bad',' poor',' Poor',' terrible',' unfavorable']]
   else:strings=[[' '+a,' '+a.capitalize()] for a in row['answers']]
   for variant,text in enumerate(row['prompts']):
    ids=tok.encode(text,add_special_tokens=False);groups=[]
    for ss in strings:
     group=[]
     for answer in ss:
      enc=tok.encode(text+answer,add_special_tokens=False)
      if len(enc)==len(ids)+1 and enc[:len(ids)]==ids:group.append(enc[-1])
     groups.append(sorted(set(group)))
    if not all(groups) or set(groups[0])&set(groups[1]):break
    ee.append(dict(text=text,ids=ids,groups=groups,variant=variant))
   if len(ee)!=2 or len(ee[0]['ids'])!=len(ee[1]['ids']):excluded.append(row|dict(reason='single-token answer or equal-length requirement'));continue
   ii=[len(examples),len(examples)+1];examples.update(zip(ii,ee));pairs.append(row|dict(indices=ii))
assert len(manifest)==64
# Historical exact prompt provenance; preserve prior benchmark prompts rather than silently moving them.
historical_files=[A/n for n in ['bilin18_expanded_task_bank.json','bilin18_context_training_bank.json','bilin18_diverse_task_expansion_screen.json','bilin18_reserved_task_format_screen.json']];historical=set()
for p in historical_files:
 for r in json.loads(p.read_text()).get('records',[]):historical.update(r.get('prompts',[]))
newtexts=[e['text'] for e in examples.values()];overlap=sorted(set(newtexts)&historical);assert not overlap and len(newtexts)==len(set(newtexts))
traintexts={examples[i]['text'] for p in pairs if p['split']=='train' for i in p['indices']};testtexts={examples[i]['text'] for p in pairs if p['split']=='combination' for i in p['indices']};assert not traintexts&testtexts
factor_audit=[]
for f in specs:
 for factor in ['content','context']:
  for value in range(4):
   counts_by_split={split:sum(p['family']==f and p[factor]==value and p['split']==split for p in pairs) for split in ['train','combination']};factor_audit.append(dict(family=f,factor=factor,value=value,**counts_by_split));assert all(v>0 for v in counts_by_split.values())
config=dict(experiment='H1053',design='Four tasks x4content pairs x4contexts. Train if(context+content)%2==0, reserve opposite parity. Every factor appears separately in training; each reserved combination is absent.',gate='Per family/split require all8aligned token-valid pairs, >=75%both-endpoint binary accuracy and>=50%semantic/exact top1. Training eligibility determined solely by train; heldout competence reported separately.',scope='Recombination of lexical content and nuisance/framing contexts, not composition of distinct computational primitives or unseen-task discovery. Semantic contrasts are task-specific and explicitly recorded.',historical_prompt_audit_sources=[str(p) for p in historical_files],historical_exact_prompt_overlap=overlap,whole_task_exclusions_preserved=['sequence_copy','spatial_inverse','role_binding','tense'],controls='Zero residual edit transfers exact donor token-content while keeping recipient residual; full boundary transplant checks target replay. Native model errors retained. No representation fitted.',excluded_pairs=excluded)
(R/f'configs/{S}.json').write_text(json.dumps(config,indent=2)+'\n');(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');mh=hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest()
model,_=facade.load_bilin18(snapshot=Path('/home/loganriggs/.cache/huggingface/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240'),device='cuda',dtype=torch.float32,verify_weights_sha256=True)
tree=ast.parse((R/'scripts/run_bilin18_finite_supervised_selectors.py').read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['suffix','KL']];exec(compile(ast.Module(body=nodes,type_ignores=[]),'<audited-native-suffix>','exec'),globals());counts={'suffix':0}
programroot=A/'content_conditioned_actions';spec=importlib.util.spec_from_file_location('content_program',programroot/'execute_content.py');program=importlib.util.module_from_spec(spec);spec.loader.exec_module(program);weights={k:v.cuda() for k,v in torch.load(programroot/'content_weights.pt',weights_only=True).items()};states={};native={};content={};contenterr=0.
with torch.no_grad():
 for i,e in examples.items():
  emb=model.transformer.wte(torch.tensor([e['ids']],device='cuda'));initial=F.rms_norm(emb,(1152,)).float();raw,x0,bus,_,_=runner.component._capture(model,initial,torch,F);states[i]=(raw,x0,bus);native[i]=suffix(states[i]).double();content[i]=program.content(weights,emb);contenterr=max(contenterr,max(float((a-b).abs().max()) for a,b in zip(content[i],states[i][1:])))
records=[];causal=[];samples=[];replay=0.
with torch.no_grad():
 for p in pairs:
  both=[];top=[];margins=[]
  for variant,i in enumerate(p['indices']):
   y=native[i];g=examples[i]['groups'];m=float(y[g[0]].logsumexp(0)-y[g[1]].logsumexp(0));margins.append(m);both.append(m*(1 if variant==0 else -1)>0);top.append(int(y.argmax()) in g[variant])
  records.append(p|dict(both_correct=all(both),top1=float(np.mean(top)),margins=margins))
  for i,d in [p['indices'],p['indices'][::-1]]:
   y=suffix((states[i][0],*content[d])).double();full=suffix(states[d]).double();replay=max(replay,float((full-native[d]).abs().max()));causal.append(dict(pair=p['id'],family=p['family'],split=p['split'],row=i,donor=d,zero_residual_KL=float(KL(native[d],y)),denominator=float(KL(native[d],native[i])),donor_top1=int(y.argmax()) in examples[i]['groups'][examples[d]['variant']]))
   if p['context']==0 and p['content']==0 and i==p['indices'][0]:samples.append(dict(family=p['family'],row=i,donor=native[d].cpu(),zero=y.cpu()))
summary=[]
for f in specs:
 for split in ['train','combination']:
  rr=[r for r in records if r['family']==f and r['split']==split];cc=[r for r in causal if r['family']==f and r['split']==split];binary=float(np.mean([r['both_correct'] for r in rr]));top=float(np.mean([r['top1'] for r in rr]));summary.append(dict(family=f,split=split,pairs=len(rr),both_endpoint_accuracy=binary,top1=top,pass_gate=len(rr)==8 and binary>=.75 and top>=.5,zero_residual_relative_KL=sum(r['zero_residual_KL'] for r in cc)/sum(r['denominator'] for r in cc)))
eligible=[r['family'] for r in summary if r['split']=='train' and r['pass_gate']];audit=dict(manifest_unchanged=mh==hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest(),factor_coverage=factor_audit,exact_train_test_prompt_overlap=0,historical_prompt_overlap_in_scanned_banks=0,content_max_abs=contenterr,full_boundary_logit_replay_max=replay,prefix_sequences=len(examples),suffix_sequences=counts['suffix'],fit_rows=0,finite=all(bool(torch.isfinite(y).all()) for y in native.values()));assert audit['manifest_unchanged'] and contenterr<1e-5 and replay<1e-5 and audit['finite']
torch.save(dict(states={i:tuple(v.cpu() for v in st) for i,st in states.items()},native={i:y.cpu() for i,y in native.items()},examples=examples),out/'states.pt');torch.save(samples,out/'audit_logits.pt');(out/'training_eligibility.json').write_text(json.dumps(dict(families=eligible,pair_ids=[r['id'] for r in records if r['family'] in eligible and r['split']=='train'],selection='Train cohort gate only; no individual success filtering or heldout outcome selection.'),indent=2)+'\n');(A/f'{S}.json').write_text(json.dumps(dict(config=config,summary=summary,records=records,causal_controls=causal,audits=audit),indent=2)+'\n');print(json.dumps(dict(summary=summary,eligible=eligible,audits={k:v for k,v in audit.items() if k!='factor_coverage'}),indent=2))

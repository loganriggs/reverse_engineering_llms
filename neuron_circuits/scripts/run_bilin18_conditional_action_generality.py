"""H1051 frozen conditional residual actions across eight nonfactual crossfit folds."""
from pathlib import Path
import sys,json,hashlib,ast,importlib.util,numpy as np,torch
from torch import nn
R=Path('/home/loganriggs/Coding/neuron_circuits');A=R/'artifacts/bilin18_tensor_decomposition';S='bilin18_conditional_action_generality';out=A/'conditional_action_generality';out.mkdir(exist_ok=True)
repo=Path('/home/loganriggs/Coding/reverse_engineer/tensor_language')
for p in [repo,repo/'basis_aligned/polynomial_causal',repo/'basis_aligned/bilinear_quotient/ops']:sys.path.insert(0,str(p))
import bilin18_observed_model_facade as facade
F=torch.nn.functional;torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');seed=10331
panel=[p for p in json.loads((A/'training_coverage_selector_comparison/panel.json').read_text())['train'] if p['family'] not in ['capital_recall','country_language']];folds=sorted({p['family'] for p in panel});assert len(folds)==8 and len(panel)==72
old=torch.load(A/'multitask_boundary_contract/states.pt',weights_only=False);added=torch.load(A/'diverse_task_expansion_screen/states.pt',weights_only=False);states={};examples={}
for p in panel:
 for i in p['indices']:
  prefix,idx=i.split(':');data=old if prefix=='old' else added;states[i]=tuple(v.cuda() for v in data['states'][int(idx)]);examples[i]=data['examples'][int(idx)]
root=A/'action_dictionary_family_crossfit';programroot=A/'content_conditioned_actions';files=[programroot/'execute_content.py',programroot/'content_weights.pt']+[root/f/n for f in folds for n in ['geometry.pt','scalar.pt','block.pt','shared.pt']];hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
config=dict(experiment='H1051',source_hashes=hashes,fit_rows=0,folds=folds,pairs=len(panel),contract='Exact token-content program supplies donor x0/cache equally to every method. Native recipient residual plus zero/sparse/dense residual change; frozen whole-family excluded encoder/decoder from H1048.',budget='2304exactcontent values/token plus max16sparse residual coordinates. Not compressed whole-state model. Retained embedding table/value matrix/native suffix counted in H1050manifest.',controls='Initial/final scalar/block/shared residual action decoders,zero residual change,dense128residual projection; no outcome selection.',scope='Competence-screened historical training-bank families used as crossfit heldout tests; native errors retained. No new task discovery or fresh external validation.')
(R/f'configs/{S}.json').write_text(json.dumps(config,indent=2)+'\n')
for file,names in [(R/'scripts/run_bilin18_finite_supervised_selectors.py',['Selector','suffix','KL']),(R/'scripts/run_bilin18_continuous_action_dictionary.py',['Selector']),(R/'scripts/run_bilin18_content_conditioned_actions.py',['residual_change'])]:
 if file.name=='run_bilin18_continuous_action_dictionary.py':OriginalSelector=Selector
 tree=ast.parse(file.read_text());nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names];exec(compile(ast.Module(body=nodes,type_ignores=[]),str(file),'exec'),globals())
spec=importlib.util.spec_from_file_location('content_program',files[0]);program=importlib.util.module_from_spec(spec);spec.loader.exec_module(program);weights={k:v.cuda() for k,v in torch.load(files[1],weights_only=True).items()}
counts={'suffix':0};model,_=facade.load_bilin18(snapshot=Path('/home/loganriggs/.cache/huggingface/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240'),device='cuda',dtype=torch.float32,verify_weights_sha256=True)
content={};native={};contenterr=0.;identityerr=0.;records=[];samples=[];maxactive=0
with torch.no_grad():
 for i,st in states.items():
  emb=model.transformer.wte(torch.tensor([examples[i]['ids']],device='cuda'));content[i]=program.content(weights,emb);contenterr=max(contenterr,max(float((a-b).abs().max()) for a,b in zip(content[i],st[1:])));native[i]=suffix(st).double();identityerr=max(identityerr,float((suffix((st[0],*content[i])).double()-native[i]).abs().max()))
 for f in folds:
  geo=torch.load(root/f/'geometry.pt',weights_only=False);B=[v.cuda() for v in geo['bases']];scale=geo['scale'].cuda();assert f not in geo['training_families']
  for kind,checkpoint in [(k,c) for k in ['scalar','block','shared'] for c in ['initial','final']]+[('dense','initial'),('zero','none')]:
   if kind not in ['dense','zero']:
    selector=Selector(kind)
    if checkpoint=='final':selector.load_state_dict(torch.load(root/f/f'{kind}.pt',weights_only=False)['state'])
   pp=[p for p in panel if p['family']==f]
   for p in pp:
    for i,d in [p['indices'],p['indices'][::-1]]:
     delta=states[d][0]-states[i][0]
     if kind=='zero':change=torch.zeros_like(delta)
     elif kind=='dense':change=(((delta.reshape(-1,1152)/scale[0])@B[0])@B[0].T*scale[0]).reshape_as(delta)
     else:change,active=residual_change(selector,delta);maxactive=max(maxactive,active)
     y=suffix((states[i][0]+change,*content[d])).double();don=native[d];base=native[i];g=examples[i]['groups'];records.append(dict(family=f,kind=kind,checkpoint=checkpoint,row=i,donor=d,paired_KL=float(KL(don,y)),paired_denominator=float(KL(don,base)),donor_top1=int(y.argmax()) in g[examples[d]['variant']]))
     if i==pp[0]['indices'][0]:samples.append(dict(family=f,kind=kind,checkpoint=checkpoint,row=i,donor=don.cpu(),paired=y.cpu()))
  print('DONE',f,flush=True)
summary=[]
for key in sorted({(r['family'],r['kind'],r['checkpoint']) for r in records}):
 rr=[r for r in records if (r['family'],r['kind'],r['checkpoint'])==key];summary.append(dict(zip(['family','kind','checkpoint'],key))|dict(paired_relative_KL=sum(r['paired_KL'] for r in rr)/sum(r['paired_denominator'] for r in rr),donor_top1=float(np.mean([r['donor_top1'] for r in rr]))))
audit=dict(source_hashes_unchanged=all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p)] for p in files),content_max_abs=contenterr,identity_logit_replay_max=identityerr,sparse_residual_active_max=maxactive,suffix_sequences=counts['suffix'],fit_rows=0,heldout_families_in_fitted_geometry=0,finite=all(np.isfinite(r['paired_KL']) for r in records));assert audit['source_hashes_unchanged'] and audit['finite'] and contenterr<1e-5 and identityerr<1e-4 and maxactive<=16
torch.save(samples,out/'audit_logits.pt');(A/f'{S}.json').write_text(json.dumps(dict(config=config,summary=summary,records=records,audits=audit),indent=2)+'\n');print(json.dumps(audit,indent=2))

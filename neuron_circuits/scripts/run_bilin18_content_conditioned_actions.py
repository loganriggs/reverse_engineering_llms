"""H1050 exact token-content program plus frozen sparse residual action controls."""
from pathlib import Path
import sys,json,hashlib,ast,importlib.util,numpy as np,torch
from torch import nn
R=Path('/home/loganriggs/Coding/neuron_circuits');A=R/'artifacts/bilin18_tensor_decomposition';S='bilin18_content_conditioned_actions';out=A/'content_conditioned_actions';out.mkdir(exist_ok=True)
repo=Path('/home/loganriggs/Coding/reverse_engineer/tensor_language')
for p in [repo,repo/'basis_aligned/polynomial_causal',repo/'basis_aligned/bilinear_quotient/ops']:sys.path.insert(0,str(p))
import bilin18_observed_model_facade as facade
import run_subject_number_sparse_graph_token_extraction_v1 as runner
F=torch.nn.functional;torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');seed=10331
source=A/'factual_cross_country_transfer';data=torch.load(source/'states.pt',weights_only=False);panel=json.loads((source/'panel.json').read_text())['pairs'];examples=data['examples'];states={i:tuple(v.cuda() for v in st) for i,st in data['states'].items()};native={i:v.cuda() for i,v in data['native'].items()};root=A/'action_dictionary_family_crossfit';folds=['capital_recall','country_language'];files=[source/'states.pt',source/'panel.json']+[root/f/n for f in folds for n in ['geometry.pt','scalar.pt','block.pt','shared.pt']];hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
config=dict(experiment='H1050',fit_rows=0,source_hashes=hashes,content='Export native firstblock full value projection and mixing scalars. Two RMS normalizations of raw pretrained token embeddings generate x0 and all9cached value heads. No attention/QK or prefix execution inside exported program.',conditional='All candidates receive identical exact content programs. Encode only residual donor-minus-recipient changes; replace embedding/cache ports with donor program outputs. Native recipient residual and native suffix retained.',cost='Not a max48total-coordinate model:2304exact content scalars pertoken plus up to16residual coordinates. Pretrained embedding lookup external and charged separately; full1152x1152value matrix retained. Active learned residual decoder147456parameters; unused other-port decoder weights not part of conditional inference.',controls='Zero residual edit, dense residual projection128, initial/final scalar/block/shared residual action decoders. Compare saved H1049all-stream compression only as unequal-information/budget reference.',scope='Same opened factual seen/excluded country panel; broad natural validation checks numerical program only, not behavioral generalization. No retraining or task discovery.')
(R/f'configs/{S}.json').write_text(json.dumps(config,indent=2)+'\n')
for file,names in [(R/'scripts/run_bilin18_finite_supervised_selectors.py',['Selector','suffix','KL']),(R/'scripts/run_bilin18_continuous_action_dictionary.py',['Selector'])]:
 if file.name=='run_bilin18_continuous_action_dictionary.py':OriginalSelector=Selector
 tree=ast.parse(file.read_text());nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names];exec(compile(ast.Module(body=nodes,type_ignores=[]),str(file),'exec'),globals())
counts={'suffix':0};model,_=facade.load_bilin18(snapshot=Path('/home/loganriggs/.cache/huggingface/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240'),device='cuda',dtype=torch.float32,verify_weights_sha256=True)
b0=model.transformer.h[0];assert b0.attn.c_v.bias is None
weights={'value':b0.attn.c_v.weight.detach().clone(),'residual0':b0.lambdas[0].detach().clone(),'embedding0':b0.lambdas[1].detach().clone()};torch.save({k:v.cpu() for k,v in weights.items()},out/'content_weights.pt')
(out/'execute_content.py').write_text('''"""Raw pretrained embeddings in; exact tokenwise x0 and cache out. No model access."""
import torch
import torch.nn.functional as F

def content(weights, embeddings):
    if embeddings.dtype != torch.float32 or embeddings.shape[-1] != 1152:
        raise ValueError("Expected float32 raw token embeddings of width1152")
    x0=F.rms_norm(embeddings,(1152,))
    raw=weights['residual0']*x0+weights['embedding0']*x0
    cache=F.linear(F.rms_norm(raw,(1152,)),weights['value'])
    return x0,cache.reshape(*cache.shape[:-1],9,128)
''')
spec=importlib.util.spec_from_file_location('exported_content',out/'execute_content.py');program=importlib.util.module_from_spec(spec);spec.loader.exec_module(program);weights={k:v.cuda() for k,v in torch.load(out/'content_weights.pt',weights_only=True).items()}
content={};sq_error=0.;sq_ref=0.;maxerr=0.;embeddingerr=0.;natural_tokens=torch.load(A/'natural_residual_basis_transfer/training_captures.pt',weights_only=False)['tokens'];unique_tokens=set(natural_tokens.flatten().tolist())
with torch.no_grad():
 for i,e in examples.items():
  emb=model.transformer.wte(torch.tensor([e['ids']],device='cuda'));x0,bus=program.content(weights,emb);content[i]=(x0,bus);embeddingerr=max(embeddingerr,float((x0-states[i][1]).abs().max()));diff=bus-states[i][2];maxerr=max(maxerr,float(diff.abs().max()));sq_error+=float(diff.double().square().sum());sq_ref+=float(states[i][2].double().square().sum())
 for start in range(0,len(natural_tokens),8):
  ids=natural_tokens[start:start+8].cuda();emb=model.transformer.wte(ids);x0,bus=program.content(weights,emb);_,native_x0,native_bus,_,_=runner.component._capture(model,F.rms_norm(emb,(1152,)).float(),torch,F);diff=bus-native_bus;maxerr=max(maxerr,float(diff.abs().max()));embeddingerr=max(embeddingerr,float((x0-native_x0).abs().max()));sq_error+=float(diff.double().square().sum());sq_ref+=float(native_bus.double().square().sum())
assert maxerr<1e-5 and embeddingerr<1e-6
@torch.no_grad()
def residual_change(selector,delta):
 c=(delta.reshape(-1,1152)/scale[0])@B[0];scores=c.square()
 if selector.kind=='scalar':mask=torch.zeros_like(c).scatter_(1,scores.topk(16,1).indices,1)
 else:
  M=selector.membership[0];gscore=scores@M.T/(4 if selector.kind=='block' else 8);g=torch.zeros_like(gscore).scatter_(1,gscore.topk(4 if selector.kind=='block' else 2,1).indices,1);mask=(g@M>0).to(c)
 return ((c*mask)@selector.decoder()[0].T*scale[0]).reshape_as(delta),float(mask.sum(1).max())
records=[];samples=[];maxactive=0.;identity_replay=0.
with torch.no_grad():
 for i in states:identity_replay=max(identity_replay,float((suffix((states[i][0],*content[i])).double()-native[i]).abs().max()))
 for f in folds:
  geo=torch.load(root/f/'geometry.pt',weights_only=False);B=[v.cuda() for v in geo['bases']];scale=geo['scale'].cuda()
  for kind,checkpoint in [(k,c) for k in ['scalar','block','shared'] for c in ['initial','final']]+[('dense','initial'),('zero','none')]:
   if kind not in ['dense','zero']:
    selector=Selector(kind)
    if checkpoint=='final':selector.load_state_dict(torch.load(root/f/f'{kind}.pt',weights_only=False)['state'])
   for p in [p for p in panel if p['family']==f]:
    for i,d in [p['indices'],p['indices'][::-1]]:
     delta=states[d][0]-states[i][0]
     if kind=='zero':change=torch.zeros_like(delta)
     elif kind=='dense':change=(((delta.reshape(-1,1152)/scale[0])@B[0])@B[0].T*scale[0]).reshape_as(delta)
     else:change,active=residual_change(selector,delta);maxactive=max(maxactive,active)
     y=suffix((states[i][0]+change,*content[d])).double();base=native[i];don=native[d];g=examples[i]['groups'];records.append(dict(family=f,cohort=p['cohort'],kind=kind,checkpoint=checkpoint,row=i,donor=d,paired_KL=float(KL(don,y)),paired_denominator=float(KL(don,base)),donor_top1=int(y.argmax()) in g[examples[d]['variant']]))
     if i==p['indices'][0] and p['id'].endswith('/0'):samples.append(dict(family=f,cohort=p['cohort'],kind=kind,checkpoint=checkpoint,row=i,donor=don.cpu(),paired=y.cpu()))
summary=[]
for key in sorted({(r['family'],r['cohort'],r['kind'],r['checkpoint']) for r in records}):
 rr=[r for r in records if (r['family'],r['cohort'],r['kind'],r['checkpoint'])==key];summary.append(dict(zip(['family','cohort','kind','checkpoint'],key))|dict(paired_relative_KL=sum(r['paired_KL'] for r in rr)/sum(r['paired_denominator'] for r in rr),donor_top1=float(np.mean([r['donor_top1'] for r in rr]))))
manifest=dict(weight_scalars=sum(v.numel() for v in weights.values()),float32_tensor_bytes=sum(v.numel()*v.element_size() for v in weights.values()),embedding_table_external_scalars=model.transformer.wte.weight.numel(),output_scalars_per_token=2304,cache_matrix_MACs_per_token=1152*1152,standalone_attention_calls=0,standalone_prefix_calls=0,files_sha256={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in ['execute_content.py','content_weights.pt']})
audit=dict(source_hashes_unchanged=all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p)] for p in files),cache_max_abs=maxerr,cache_relative_L2=(sq_error/sq_ref)**.5,embedding_max_abs=embeddingerr,natural_documents=len(natural_tokens),natural_token_occurrences=natural_tokens.numel(),natural_unique_tokens=len(unique_tokens),natural_prefix_sequences=len(natural_tokens),suffix_sequences=counts['suffix'],content_identity_logit_replay_max=identity_replay,sparse_residual_active_max=maxactive,fit_rows=0,finite=all(np.isfinite(r['paired_KL']) for r in records));assert audit['source_hashes_unchanged'] and audit['finite'] and identity_replay<1e-4 and maxactive<=16
(out/'content_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');torch.save(samples,out/'audit_logits.pt');(A/f'{S}.json').write_text(json.dumps(dict(config=config,manifest=manifest,summary=summary,records=records,audits=audit),indent=2)+'\n');print(json.dumps(dict(manifest=manifest,audits=audit),indent=2))

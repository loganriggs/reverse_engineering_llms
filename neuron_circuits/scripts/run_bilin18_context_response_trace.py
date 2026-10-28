"""H801 cross native surrounding context and L2 pre-MLP source-local state."""
import json,time,re,math
from pathlib import Path
from collections import defaultdict
import torch
from transformers import AutoTokenizer
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_broad_task_competence import continuation_id
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_reader_operation_format import run
from run_bilin18_coactivation_group_swaps import sha
S='bilin18_context_response_trace'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'capture':ART/'bilin18_native_key_value_cross_captures.pt','manifest':ART/'bilin18_native_key_value_cross_manifest.json','reference':ART/'bilin18_fixed_value_context_response.json','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};cap=torch.load(paths['capture'],weights_only=False);manifest={r['id']:r for r in json.loads(paths['manifest'].read_text())};reference=json.loads(paths['reference'].read_text());cfg={'trace_design':'Trace attention, MLP outputs and block residuals at source1, source2, final query in L2-L17. Compare perturbation minus own hybrid baseline across contexts at fixed local state. Descriptive nomination, not causal mediation.','experiment':'H801','hashes':hashes,'deltas':[0.,.5],'design':'Cross surrounding same/different-key prompt with native same/different-key second-source pre-MLP L2 residual state. Add identical delta*W_h*V at that residual. Compare each perturbation to its own crossed baseline. Freeze item/format/query pairing.','prediction':'Response follows transplanted local state if local geometry explains conditioning, or surrounding context if later readers/other token states matter; interaction between them is possible.','limitations':'Hybrid local-state substitutions are diagnostic, not guaranteed on-manifold. State swap changes the entire local residual, not a semantic gate. Other token states/native context unchanged.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);examples={};lookup={}
 for rid,c in cap.items():
  r=manifest[rid]
  if not r['same_value']:continue
  enc=tok(r['text'],add_special_tokens=False,return_offsets_mapping=True);spans=[m.span() for m in re.finditer(r'\b'+r['values'][0]+r'\b',r['text'])];pp=[]
  for a,b in spans:
   ix=[i for i,(x,y) in enumerate(enc['offset_mapping']) if x<b and y>a];assert len(ix)==1;pp.append(ix[0])
  target=continuation_id(tok,r['text'],' '+r['values'][0]);assert target is not None;examples[rid]={**r,'ids':enc['input_ids'],'positions':pp,'target':target};lookup[r['format'],r['item'],r['query'],r['same_key']]=rid
 model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 W=model.blocks[2].attn.c_proj.weight[:,640:768].cpu().double();directions={rid:W@cap[rid]['factors'][4].double() for rid in examples};directionerr=[]
 for rid,e in examples.items():
  other=lookup[e['format'],e['item'],e['query'],not e['same_key']];directionerr.append(float((directions[rid]-directions[other]).abs().max()));assert examples[other]['positions']==e['positions']
 native={};local={};saved={};batch=[];forwards=0
 def blockpre(m,args):saved['raw']=m.lambdas[0]*args[0]+m.lambdas[1]*args[2]
 def native_attn(m,args,out):saved['local']=torch.stack([(saved['raw']+out[0])[i,e['positions'][1]] for i,e in enumerate(batch)]).cpu()
 hs=[model.blocks[2].register_forward_pre_hook(blockpre),model.blocks[2].attn.register_forward_hook(native_attn)];buckets=defaultdict(list)
 for rid,e in examples.items():buckets[len(e['ids'])].append(rid)
 try:
  for rr in buckets.values():
   for lo in range(0,len(rr),12):
    bb=rr[lo:lo+12];batch=[examples[k] for k in bb];out,_=run(model,torch.tensor([e['ids'] for e in batch],device='cuda'));forwards+=len(bb)
    for i,k in enumerate(bb):native[k]=out[i].cpu();local[k]=saved['local'][i]
 finally:
  for h in hs:h.remove()
 jobs=[{'id':rid,'state_key':sk,'delta':delta} for rid in examples for sk in [False,True] for delta in cfg['deltas']];buckets=defaultdict(list)
 for j in jobs:buckets[len(examples[j['id']]['ids'])].append(j)
 outputs={};replacement=[];identity=[];traces={};trace_batch={}
 for jj in buckets.values():
  for lo in range(0,len(jj),12):
   bb=jj[lo:lo+12]
   def clamp(m,args,out):
    y=out[0].clone()
    for i,j in enumerate(bb):
     e=examples[j['id']];don=lookup[e['format'],e['item'],e['query'],j['state_key']];p=e['positions'][1];desired=(local[don].double()+j['delta']*directions[j['id']]).to(y);y[i,p]=desired-saved['raw'][i,p];replacement.append(float(((saved['raw'][i,p]+y[i,p])-desired).norm()/desired.norm()))
    return(y,out[1])
   hs=[model.blocks[2].register_forward_pre_hook(blockpre),model.blocks[2].attn.register_forward_hook(clamp)]
   trace_batch={}
   for layer in range(2,18):
    for stage,module in [('attention',model.blocks[layer].attn),('mlp',model.blocks[layer].mlp),('residual',model.blocks[layer])]:
     def trace(m,args,out,layer=layer,stage=stage):
      y=out if stage=='mlp' else out[0]
      trace_batch[layer,stage]=torch.stack([y[i,[examples[j['id']]['positions'][0],examples[j['id']]['positions'][1],len(examples[j['id']]['ids'])-1]] for i,j in enumerate(bb)]).cpu()
     hs.append(module.register_forward_hook(trace))
   try:out,_=run(model,torch.tensor([examples[j['id']]['ids'] for j in bb],device='cuda'))
   finally:
    for h in hs:h.remove()
   forwards+=len(bb)
   for i,j in enumerate(bb):
    outputs[j['id'],j['state_key'],j['delta']]=out[i].cpu()
    traces[j['id'],j['state_key'],j['delta']]={k:v[i].clone() for k,v in trace_batch.items()}
    if j['delta']==0 and j['state_key']==examples[j['id']]['same_key']:identity.append(float((out[i].cpu()-native[j['id']]).abs().max()))
 old={(r['id'],r['scale']):r for r in reference['records']};records=[];replay=[]
 for j in jobs:
  rid=j['id'];e=examples[rid];p=outputs[rid,j['state_key'],j['delta']].double().log_softmax(0);base=outputs[rid,j['state_key'],0.].double().log_softmax(0);nat=native[rid].double().log_softmax(0);target=e['target'];change=float(p[target]-base[target]);records.append({'id':rid,'item':e['item'],'format':e['format'],'query':e['query'],'context_same_key':e['same_key'],'state_same_key':j['state_key'],'delta':j['delta'],'intended_logprob_response':change,'hybrid_baseline_logprob_change_from_native':float(base[target]-nat[target]),'KL_hybrid_to_perturbed':float((base.exp()*(base-p)).sum()),'KL_native_to_hybrid_baseline':float((nat.exp()*(nat-base)).sum()),'hybrid_baseline_full_correct':int(base.argmax())==target,'perturbed_full_correct':int(p.argmax())==target})
  if j['state_key']==e['same_key']:replay.append(abs(change-old[rid,j['delta']]['intended_logprob_change']))
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'matched_projected_direction_max_abs':max(directionerr),'local_residual_clamp_max_relative_L2':max(replacement),'native_state_zero_delta_logits_max_abs':max(identity),'native_state_response_H799_max_abs':max(replay),'finite':all(math.isfinite(r['intended_logprob_response']) for r in records),'whole_model_forwards':forwards,'test_evaluations':0};assert audits['hashes_unchanged'] and audits['finite'] and max(directionerr)<1e-5 and max(replacement)<1e-5 and max(identity+replay)<1e-4
 torch.save({'traces':traces,'examples':examples,'local':local,'directions':directions},ART/f'{S}_captures.pt');(ART/f'{S}.json').write_text(json.dumps({'status':'complete','config':cfg,'records':records,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(audits)
if __name__=='__main__':main()

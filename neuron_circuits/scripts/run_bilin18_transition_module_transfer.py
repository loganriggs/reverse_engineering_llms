"""H805 cross native surrounding context and L2 pre-MLP source-local state."""
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
S='bilin18_transition_module_transfer'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'capture':ART/'bilin18_native_key_value_cross_captures.pt','manifest':ART/'bilin18_native_key_value_cross_manifest.json','reference':ART/'bilin18_fixed_value_context_response.json','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};paths['trace']=ART/'bilin18_context_response_trace_captures.pt';hashes['trace']=sha(paths['trace']);tr=torch.load(paths['trace'],weights_only=False)['traces'];cap=torch.load(paths['capture'],weights_only=False);manifest={r['id']:r for r in json.loads(paths['manifest'].read_text())};reference=json.loads(paths['reference'].read_text());cfg={'response_transfer':'At L5 or L6 final query, swap attention/MLP/both module perturbation responses across contexts at fixed local source state. Retain recipient zero-delta module baseline. Joint arm clamps both outputs using original captured responses, not a recomputed MLP response after attention intervention.','experiment':'H805','hashes':hashes,'deltas':[0.,.5],'design':'Cross surrounding same/different-key prompt with native same/different-key second-source pre-MLP L2 residual state. Add identical delta*W_h*V at that residual. Compare each perturbation to its own crossed baseline. Freeze item/format/query pairing.','prediction':'Response follows transplanted local state if local geometry explains conditioning, or surrounding context if later readers/other token states matter; interaction between them is possible.','limitations':'Hybrid local-state substitutions are diagnostic, not guaranteed on-manifold. State swap changes the entire local residual, not a semantic gate. Other token states/native context unchanged.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);examples={};lookup={}
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
 jobs=[{'id':rid,'state_key':sk,'delta':delta,'arm':arm,'layer':layer} for rid in examples for sk in [False,True] for delta in cfg['deltas'] for arm in ['none','attention','mlp','both'] for layer in [5,6]];buckets=defaultdict(list)
 for j in jobs:buckets[len(examples[j['id']]['ids'])].append(j)
 outputs={};replacement=[];identity=[]
 for jj in buckets.values():
  for lo in range(0,len(jj),12):
   bb=jj[lo:lo+12]
   def clamp(m,args,out):
    y=out[0].clone()
    for i,j in enumerate(bb):
     e=examples[j['id']];don=lookup[e['format'],e['item'],e['query'],j['state_key']];p=e['positions'][1];desired=(local[don].double()+j['delta']*directions[j['id']]).to(y);y[i,p]=desired-saved['raw'][i,p];replacement.append(float(((saved['raw'][i,p]+y[i,p])-desired).norm()/desired.norm()))
    return(y,out[1])
   hs=[model.blocks[2].register_forward_pre_hook(blockpre),model.blocks[2].attn.register_forward_hook(clamp)]
   for layer in [5,6]:
    for stage,module in [('attention',model.blocks[layer].attn),('mlp',model.blocks[layer].mlp)]:
     def transfer(m,args,out,layer=layer,stage=stage):
      y=(out[0] if stage=='attention' else out).clone()
      for i,j in enumerate(bb):
       if j['layer']!=layer or j['arm'] not in [stage,'both']:continue
       rid=j['id'];e=examples[rid];sk=j['state_key'];other=lookup[e['format'],e['item'],e['query'],not e['same_key']];p=len(e['ids'])-1
       base=tr[rid,sk,0.][layer,stage][2].double()
       response=tr[other,sk,j['delta']][layer,stage][2].double()-tr[other,sk,0.][layer,stage][2].double()
       y[i,p]=(base+response).to(y)
      return (y,out[1]) if stage=='attention' else y
     hs.append(module.register_forward_hook(transfer))
   try:out,_=run(model,torch.tensor([examples[j['id']]['ids'] for j in bb],device='cuda'))
   finally:
    for h in hs:h.remove()
   forwards+=len(bb)
   for i,j in enumerate(bb):
    outputs[j['id'],j['state_key'],j['delta'],j['arm'],j['layer']]=out[i].cpu()
    if j['delta']==0 and j['state_key']==examples[j['id']]['same_key']:identity.append(float((out[i].cpu()-native[j['id']]).abs().max()))
 old={(r['id'],r['scale']):r for r in reference['records']};records=[];replay=[]
 for j in jobs:
  rid=j['id'];e=examples[rid];p=outputs[rid,j['state_key'],j['delta'],j['arm'],j['layer']].double().log_softmax(0);base=outputs[rid,j['state_key'],0.,j['arm'],j['layer']].double().log_softmax(0);nat=native[rid].double().log_softmax(0);target=e['target'];change=float(p[target]-base[target]);records.append({'layer':j['layer'],'arm':j['arm'],'id':rid,'item':e['item'],'format':e['format'],'query':e['query'],'context_same_key':e['same_key'],'state_same_key':j['state_key'],'delta':j['delta'],'intended_logprob_response':change,'hybrid_baseline_logprob_change_from_native':float(base[target]-nat[target]),'KL_hybrid_to_perturbed':float((base.exp()*(base-p)).sum()),'KL_native_to_hybrid_baseline':float((nat.exp()*(nat-base)).sum()),'hybrid_baseline_full_correct':int(base.argmax())==target,'perturbed_full_correct':int(p.argmax())==target})
  if j['arm']=='none' and j['state_key']==e['same_key']:replay.append(abs(change-old[rid,j['delta']]['intended_logprob_change']))
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'matched_projected_direction_max_abs':max(directionerr),'local_residual_clamp_max_relative_L2':max(replacement),'native_state_zero_delta_logits_max_abs':max(identity),'native_state_response_H799_max_abs':max(replay),'finite':all(math.isfinite(r['intended_logprob_response']) for r in records),'whole_model_forwards':forwards,'test_evaluations':0};assert audits['hashes_unchanged'] and audits['finite'] and max(directionerr)<1e-5 and max(replacement)<1e-5 and max(identity+replay)<1e-4
 torch.save({'local':local,'directions':directions},ART/f'{S}_captures.pt');(ART/f'{S}.json').write_text(json.dumps({'status':'complete','config':cfg,'records':records,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(audits)
if __name__=='__main__':main()

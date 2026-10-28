"""H803 cross native surrounding context and L2 pre-MLP source-local state."""
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
S='bilin18_depth_response_transfer'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'capture':ART/'bilin18_native_key_value_cross_captures.pt','manifest':ART/'bilin18_native_key_value_cross_manifest.json','reference':ART/'bilin18_fixed_value_context_response.json','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};cap=torch.load(paths['capture'],weights_only=False);manifest={r['id']:r for r in json.loads(paths['manifest'].read_text())};reference=json.loads(paths['reference'].read_text());cfg={'experiment':'H803','hashes':hashes,'deltas':[0.,.5],'design':'Cross surrounding same/different-key prompt with native same/different-key second-source pre-MLP L2 residual state. Add identical delta*W_h*V at that residual. Compare each perturbation to its own crossed baseline. Freeze item/format/query pairing.','prediction':'Response follows transplanted local state if local geometry explains conditioning, or surrounding context if later readers/other token states matter; interaction between them is possible.','limitations':'Hybrid local-state substitutions are diagnostic, not guaranteed on-manifold. State swap changes the entire local residual, not a semantic gate. Other token states/native context unchanged.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);examples={};lookup={}
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
 cuts=[3,8,13,17]
 cfg['cuts']=cuts;cfg['design']='Transfer other-context perturbation-minus-baseline residual response at a depth cut, while preserving recipient baseline. Compare all-token versus query-only; both local source states, both delta0 and .5.'
 cfg['hypotheses_ranked_before_results']=[
 ['high','Later cross-token readers regenerate context dependence.'],
 ['high','Final readout responds differently to the same residual response on different baselines.'],
 ['high','Multiple token paths bypass the two L3 sites patched in H802.'],
 ['medium','A later MLP transforms source content conditionally.'],
 ['medium','Query-state geometry gates a shared source signal.'],
 ['medium','RMS normalization accounts for a substantial part of baseline-conditioned readout.'],
 ['medium','Unembedding competitors, rather than intended-logit change, drive the logprob contrast.'],
 ['medium','Perturbation amplitude crosses a nonlinear response threshold.'],
 ['medium','Context changes response direction more than its norm.'],
 ['medium','Context changes response norm more than its direction.'],
 ['medium','Residual reinjection of original embeddings regenerates conditioning after a cut.'],
 ['medium','Persistent L0 value bus provides a parallel context route.'],
 ['medium','Same-key repeated queries pool distinct reader behaviors.'],
 ['medium','Finite hybrid interventions leave the native manifold and exaggerate differences.'],
 ['medium','Lexical or formatting features explain effects without semantic key equality.'],
 ['low after H800','L2 source-local state alone explains the contrast.'],
 ['low after H802','The earliest L3 source/query attention differences explain most output contrast.'],
 ['low after H799','Unequal injected value direction or magnitude explains context difference.'],
 ['open','Shared dictionary atoms transfer content but require separate contextual gates.'],
 ['open','A response-family representation is more faithful than static manifold labels.']]
 (ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 jobs=[{'id':rid,'state':sk,'delta':delta} for rid in examples for sk in [False,True] for delta in cfg['deltas']]
 captures={};logits={};tracebatch={};clamp_err=[]
 def local_clamp(m,args,out):
  y=out[0].clone()
  for i,j in enumerate(bb):
   e=examples[j['id']];don=lookup[e['format'],e['item'],e['query'],j['state']];p=e['positions'][1];desired=(local[don].double()+j['delta']*directions[j['id']]).to(y);y[i,p]=desired-saved['raw'][i,p];clamp_err.append(float(((saved['raw'][i,p]+y[i,p])-desired).norm()/desired.norm()))
  return y,out[1]
 buckets=defaultdict(list)
 for j in jobs:buckets[len(examples[j['id']]['ids'])].append(j)
 for jj in buckets.values():
  for lo in range(0,len(jj),12):
   bb=jj[lo:lo+12];tracebatch={};hs=[model.blocks[2].register_forward_pre_hook(blockpre),model.blocks[2].attn.register_forward_hook(local_clamp)]
   for layer in cuts:
    def capture(m,args,out,layer=layer):tracebatch[layer]=out[0].cpu()
    hs.append(model.blocks[layer].register_forward_hook(capture))
   try:out,_=run(model,torch.tensor([examples[j['id']]['ids'] for j in bb],device='cuda'))
   finally:
    for h in hs:h.remove()
   forwards+=len(bb)
   for i,j in enumerate(bb):
    k=j['id'],j['state'],j['delta'];logits[k]=out[i].cpu();captures[k]={l:v[i].clone() for l,v in tracebatch.items()}
 records=[];baseline_err=[];final_query_all={};old={(r['id'],r['state_same_key'],r['delta']):r for r in json.loads((ART/'bilin18_local_state_context_cross.json').read_text())['records']};replay=[]
 def measure(j,out,cut,site):
  rid=j['id'];e=examples[rid];sk=j['state'];delta=j['delta'];p=out.double().log_softmax(0);base=logits[rid,sk,0.].double().log_softmax(0);t=e['target'];response=float(p[t]-base[t]);records.append({'id':rid,'format':e['format'],'item':e['item'],'query':e['query'],'context_same_key':e['same_key'],'state_same_key':sk,'delta':delta,'cut':cut,'site':site,'response':response,'KL_base_to_output':float((base.exp()*(base-p)).sum()),'correct':int(p.argmax())==t})
  if cut==-1:replay.append(abs(response-old[rid,sk,delta]['intended_logprob_response']))
 for j in jobs:measure(j,logits[j['id'],j['state'],j['delta']],-1,'none')
 for layer in cuts:
  for site in ['query','all']:
   for jj in buckets.values():
    for lo in range(0,len(jj),12):
     bb=jj[lo:lo+12]
     def transfer(m,args,out):
      y=out[0].clone()
      for i,j in enumerate(bb):
       rid=j['id'];e=examples[rid];sk=j['state'];other=lookup[e['format'],e['item'],e['query'],not e['same_key']];base=captures[rid,sk,0.][layer].double();response=captures[other,sk,j['delta']][layer].double()-captures[other,sk,0.][layer].double();assert base.shape==response.shape
       desired=(base+response).to(y)
       if site=='all':y[i]=desired
       else:y[i,-1]=desired[-1]
      return y,out[1]
     hs=[model.blocks[2].register_forward_pre_hook(blockpre),model.blocks[2].attn.register_forward_hook(local_clamp),model.blocks[layer].register_forward_hook(transfer)]
     try:out,_=run(model,torch.tensor([examples[j['id']]['ids'] for j in bb],device='cuda'))
     finally:
      for h in hs:h.remove()
     forwards+=len(bb)
     for i,j in enumerate(bb):
      measure(j,out[i].cpu(),layer,site)
      if j['delta']==0.:baseline_err.append(float((out[i].cpu()-logits[j['id'],j['state'],0.]).abs().max()))
      if layer==17:final_query_all[j['id'],j['state'],j['delta'],site]=out[i].cpu()
   print('finished',layer,site,flush=True)
 finalerr=max(float((v-final_query_all[rid,sk,de,'all']).abs().max()) for (rid,sk,de,si),v in final_query_all.items() if si=='query')
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'matched_direction_max_abs':max(directionerr),'local_clamp_max_relative':max(clamp_err),'zero_delta_transfer_full_logits_max_abs':max(baseline_err),'H800_response_max_abs':max(replay),'final_all_query_full_logits_max_abs':finalerr,'finite':all(math.isfinite(r['response']) for r in records),'whole_model_forwards':forwards,'test_evaluations':0};assert audits['hashes_unchanged'] and audits['finite'] and max(baseline_err+replay+[finalerr])<1e-4
 torch.save({'captures':captures,'logits':logits,'examples':examples},ART/f'{S}_captures.pt');(ART/f'{S}.json').write_text(json.dumps({'config':cfg,'records':records,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(audits)
if __name__=='__main__':main()

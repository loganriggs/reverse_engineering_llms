"""H807 cross native surrounding context and L2 pre-MLP source-local state."""
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
S='bilin18_mlp_input_baseline_fresh'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'capture':ART/'bilin18_fresh_bird_capture_captures.pt','manifest':ART/'bilin18_fresh_bird_capture_manifest.json','reference':ART/'bilin18_fixed_value_context_response.json','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin'};paths['lock']=ROOT/'configs/bilin18_mlp_input_baseline_fresh_locked.json';hashes={k:sha(p) for k,p in paths.items()};cap=torch.load(paths['capture'],weights_only=False);manifest={r['id']:r for r in json.loads(paths['manifest'].read_text())};reference=json.loads(paths['reference'].read_text());cfg={'experiment':'H807','hashes':hashes,'deltas':[0.,.5],'design':'Cross surrounding same/different-key prompt with native same/different-key second-source pre-MLP L2 residual state. Add identical delta*W_h*V at that residual. Compare each perturbation to its own crossed baseline. Freeze item/format/query pairing.','prediction':'Response follows transplanted local state if local geometry explains conditioning, or surrounding context if later readers/other token states matter; interaction between them is possible.','limitations':'Hybrid local-state substitutions are diagnostic, not guaranteed on-manifold. State swap changes the entire local residual, not a semantic gate. Other token states/native context unchanged.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);examples={};lookup={}
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
 import torch.nn.functional as F
 cfg['design']='Cross L6 query raw pre-MLP baseline and finite input response from paired contexts; RMS normalize each endpoint, run exact MLP, transplant only resulting module response onto recipient module baseline. Four cells own/input_swap/baseline_swap/both. Surrounding recipient and incoming skip response unchanged.'
 (ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 jobs=[{'id':rid,'state':sk,'delta':de} for rid in examples for sk in [False,True] for de in [0.,.25,.5]];buckets=defaultdict(list)
 for j in jobs:buckets[len(examples[j['id']]['ids'])].append(j)
 captures={};logits={};clamp_err=[];normerr=[]
 def local_clamp(m,args,out):
  y=out[0].clone()
  for i,j in enumerate(bb):
   e=examples[j['id']];don=lookup[e['format'],e['item'],e['query'],j['state']];p=e['positions'][1];desired=(local[don].double()+j['delta']*directions[j['id']]).to(y);y[i,p]=desired-saved['raw'][i,p];clamp_err.append(float(((saved['raw'][i,p]+y[i,p])-desired).norm()/desired.norm()))
  return y,out[1]
 def pre6(m,args):saved['raw6']=m.lambdas[0]*args[0]+m.lambdas[1]*args[2]
 def att6(m,args,out):saved['premlp']=(saved['raw6']+out[0])[:,-1].clone()
 def mlppre(m,args):saved['normalized']=args[0][:,-1].clone()
 def mlpout(m,args,out):saved['mlpout']=out[:,-1].clone()
 for jj in buckets.values():
  for lo in range(0,len(jj),12):
   bb=jj[lo:lo+12];hs=[model.blocks[2].register_forward_pre_hook(blockpre),model.blocks[2].attn.register_forward_hook(local_clamp),model.blocks[6].register_forward_pre_hook(pre6),model.blocks[6].attn.register_forward_hook(att6),model.blocks[6].mlp.register_forward_pre_hook(mlppre),model.blocks[6].mlp.register_forward_hook(mlpout)]
   try:out,_=run(model,torch.tensor([examples[j['id']]['ids'] for j in bb],device='cuda'))
   finally:
    for h in hs:h.remove()
   forwards+=len(bb);normerr.append(float((F.rms_norm(saved['premlp'],(1152,))-saved['normalized']).abs().max()))
   for i,j in enumerate(bb):
    k=j['id'],j['state'],j['delta'];captures[k]={n:saved[n][i].cpu() for n in ['premlp','normalized','mlpout']};logits[k]=out[i].cpu()
 mlp=model.blocks[6].mlp;assert not mlp.config.gated
 L,R,D=[getattr(mlp,n).weight.double() for n in ['Left','Right','Down']];targets={};decomposition=[];reconstruction=[];exacterr=[]
 for rid,e in examples.items():
  other=lookup[e['format'],e['item'],e['query'],not e['same_key']]
  for sk in [False,True]:
   for amp in [.25,.5]:
    for arm in ['own','input_swap','baseline_swap','both']:
     bk=other if arm in ['baseline_swap','both'] else rid;dk=other if arm in ['input_swap','both'] else rid
     b=captures[bk,sk,0.]['premlp'].cuda();dx=(captures[dk,sk,amp]['premlp']-captures[dk,sk,0.]['premlp']).cuda()
     u=F.rms_norm(b,(1152,));v=F.rms_norm(b+dx,(1152,));du=(v-u).double();ud=u.double()
     direct=mlp(v)-mlp(u);lin=F.linear(F.linear(ud,L)*F.linear(du,R)+F.linear(du,L)*F.linear(ud,R),D);quad=F.linear(F.linear(du,L)*F.linear(du,R),D);expanded=lin+quad
     exacterr.append(float((direct.double()-expanded).norm()/direct.norm().clamp_min(1e-20)))
     if arm in ['own','both']:
      expected=captures[bk,sk,amp]['mlpout']-captures[bk,sk,0.]['mlpout'];reconstruction.append(float((direct.cpu()-expected).norm()/expected.norm().clamp_min(1e-20)))
     targets[rid,sk,arm,amp]=direct.cpu();decomposition.append({'id':rid,'format':e['format'],'item':e['item'],'state':sk,'arm':arm,'amplitude':amp,'response_norm':float(direct.norm()),'linear_norm':float(lin.norm()),'quadratic_norm':float(quad.norm()),'quadratic_to_response':float(quad.norm()/expanded.norm().clamp_min(1e-20))})
 records=[];identity=[];replay=[]
 for arm in ['own','input_swap','baseline_swap','both']:
  for jj in buckets.values():
   for lo in range(0,len(jj),12):
    bb=jj[lo:lo+12]
    def patch(m,args,out):
     y=out.clone()
     for i,j in enumerate(bb):
      rid=j['id'];sk=j['state'];base=captures[rid,sk,0.]['mlpout'];response=targets[rid,sk,arm,j['delta']] if j['delta'] else torch.zeros_like(base);y[i,-1]=(base+response).to(y)
     return y
    hs=[model.blocks[2].register_forward_pre_hook(blockpre),model.blocks[2].attn.register_forward_hook(local_clamp),model.blocks[6].mlp.register_forward_hook(patch)]
    try:out,_=run(model,torch.tensor([examples[j['id']]['ids'] for j in bb],device='cuda'))
    finally:
     for h in hs:h.remove()
    forwards+=len(bb)
    for i,j in enumerate(bb):
     rid=j['id'];sk=j['state'];de=j['delta'];e=examples[rid];p=out[i].cpu().double().log_softmax(0);base=logits[rid,sk,0.].double().log_softmax(0);change=float(p[e['target']]-base[e['target']]);records.append({'id':rid,'format':e['format'],'item':e['item'],'query':e['query'],'context_same_key':e['same_key'],'state':sk,'delta':de,'arm':arm,'response':change,'probability_change':float(p[e['target']].exp()-base[e['target']].exp()),'KL_base_to_output':float((base.exp()*(base-p)).sum()),'correct':int(p.argmax())==e['target']})
     if de==0:identity.append(float((out[i].cpu()-logits[rid,sk,0.]).abs().max()))
     if arm=='own':replay.append(float((out[i].cpu()-logits[rid,sk,de]).abs().max()))
  print('finished',arm,flush=True)
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'RMS_input_reconstruction_max_abs':max(normerr),'bilinear_expansion_max_relative':max(exacterr),'direct_module_response_reconstruction_max_relative':max(reconstruction),'zero_delta_full_logits_max_abs':max(identity),'own_reconstructed_full_logits_max_abs':max(replay),'local_clamp_relative':max(clamp_err),'finite':all(math.isfinite(r['response']) for r in records),'whole_model_forwards':forwards,'test_evaluations':0};assert audits['hashes_unchanged'] and audits['finite'] and max(normerr+exacterr+reconstruction+identity+replay)<1e-4
 torch.save({'captures':captures,'targets':targets,'examples':examples},ART/f'{S}_captures.pt');(ART/f'{S}.json').write_text(json.dumps({'config':cfg,'records':records,'decomposition':decomposition,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(audits)
if __name__=='__main__':main()

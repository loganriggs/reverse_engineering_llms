#!/usr/bin/env python3
"""H542: matched-token/position query-key context factorial, with live downstream computation."""
import json,time,hashlib
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_qkov_causal_edges import load_full_model
STEM='bilin18_reader_query_key_context';HEADS=[(14,4),(8,3),(14,0),(8,0)]
torch.set_num_threads(6)
@torch.inference_mode()
def run(model,ids,layer=None,head=None,source_pos=None,query=None,key=None,delta=None):
 d=model.config.n_embd;hd=d//model.config.n_head;x0=F.rms_norm(model.embedding(ids),(d,));x=x0;bus=None;count=0
 for l,b in enumerate(model.blocks):
  handles=[]
  if l==layer:
   for n in ('c_q','c_q2','c_k','c_k2'):
    def hook(m,args,out,n=n):
     nonlocal count
     out=out.clone();sl=slice(head*hd,(head+1)*hd)
     if n in ('c_q','c_q2'):out[:,-1,sl]=query[n].to(out)
     else:out[:,source_pos,sl]=key[n].to(out)
     count+=1;return out
    handles.append(getattr(b.attn,n).register_forward_hook(hook))
   if delta is not None:
    def ahook(m,args,out):
     y,v=out;y=y.clone();y[:,-1]+=delta.to(y);return y,v
    handles.append(b.attn.register_forward_hook(ahook))
  try:x,bus=b(x,bus,x0)
  finally:
   for h in handles:h.remove()
 if layer is not None:assert count==4
 return 30*torch.tanh(model.lm_head(F.rms_norm(x,(d,))[:,-1]).float()/30)
@torch.inference_mode()
def main():
 start=time.time();src=ART/'bilin18_reader_lexical_role_crossing.json';old=json.loads(src.read_text());cp=ART/'bilin18_reader_lexical_role_crossing_native_cache.pt';saved=torch.load(cp,weights_only=False,map_location='cpu');cache,examples=saved['cache'],saved['examples'];locked=json.loads((ART/'bilin18_reader_identity_transfer_locked_signs.json').read_text())
 cfg={'experiment':'H542','source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'native_cache_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),'heads':HEADS,'contexts':['binding standard','copying position_matched'],'factorial':'Query context B/C x Key context B/C, at same source token, final token and absolute positions. Both query projections copied at final token of one head; both key projections at matched source position of that head. All other projections native.','scope':'Actual Q/K swaps can affect other edges of that head and other destinations. A third-token persistent-bus change is additionally applied only to the designated source/head/final-destination edge using its recombined coefficient. Downstream computations live.','third_prediction':'H539 discovery identity-map sign times recombined coefficient sign.','selection':'No new fits or head selection. Reused H541 matched contexts; diagnostic, not independent fresh confirmation.','hypotheses':['Contextual source keys can select the reader sign independently of final-token query context.','Alternatively query context selects the sign, or a query/key interaction is required.','The locked identity rule survives actual Q/K swaps.'],'precision':'float32 compute after shared-loader bfloat16 rounding'}
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2)+'\n');print('LOCKED',flush=True)
 model=load_full_model('cuda')
 for mod in (model.embedding,model.blocks,model.lm_head):
  mod.float().eval()
  for p in mod.parameters():p.requires_grad_(False)
 H=model.config.n_head;D=model.config.n_embd//H;result={'config':cfg,'matching':[],'geometry':[],'interventions':[],'third_comparisons':[],'status':'running'};out=ART/f'{STEM}.json';jobs=defaultdict(list);native={}
 def save():result['elapsed_seconds']=time.time()-start;out.write_text(json.dumps(result,indent=2)+'\n')
 def token_bus(token):
  x0=F.rms_norm(model.embedding(torch.tensor([token],device='cuda')),(model.config.n_embd,));x=model.blocks[0].lambdas.sum()*x0;return model.blocks[0].attn.c_v(F.rms_norm(x,(model.config.n_embd,))).reshape(H,D)
 for i in range(8):
  for variant in (0,1):
   ids={'B':(f'matched-binding-standard-{i}',variant),'C':(f'matched-copy-position_matched-{i}',variant)};ee={ctx:examples[k] for ctx,k in ids.items()};cc={ctx:cache[k] for ctx,k in ids.items()};T=len(ee['B']['ids']);positions=[p for p,(a,b) in enumerate(zip(ee['B']['ids'],examples[(ids['B'][0],1-variant)]['ids'])) if a!=b];assert len(positions)==1;p=positions[0]
   assert T==len(ee['C']['ids']) and ee['B']['ids'][-1]==ee['C']['ids'][-1] and ee['B']['ids'][p]==ee['C']['ids'][p]
   before=ee['B']['ids'][p];after=examples[(ids['B'][0],1-variant)]['ids'][p];third=ee['B']['third'];assert third==ee['C']['third']
   result['matching'].append({'item':i,'variant':variant,'source_position':p,'final_position':T-1,'source_token':before,'final_token':ee['B']['ids'][-1]})
   for ctx in ('B','C'):
    logits=run(model,torch.tensor([ee[ctx]['ids']],device='cuda'));native[(i,variant,ctx)]=measure(logits[0],[[before],[after]])['margin']
   for l,h in HEADS:
    attn=model.blocks[l].attn;vectors={}
    for ctx in ('B','C'):
     raw={n:cc[ctx][(l,n)].cuda().view(1,T,H,D) for n in ('c_q','c_q2','c_k','c_k2')};cos,sin=attn.rotary(raw['c_q']);vectors[ctx]={n:model.tt.apply_rotary_emb(F.rms_norm(x,(D,)),cos,sin)[0,-1 if n in ('c_q','c_q2') else p,h] for n,x in raw.items()}
    for qctx in ('B','C'):
     for kctx in ('B','C'):
      f1=float(vectors[qctx]['c_q']@vectors[kctx]['c_k']/D);f2=float(vectors[qctx]['c_q2']@vectors[kctx]['c_k2']/D);coef=f1*f2*float(attn.lamb);result['geometry'].append({'item':i,'variant':variant,'layer':l,'head':h,'query_context':qctx,'key_context':kctx,'factor1':f1,'factor2':f2,'coefficient':coef})
      qq={n:cc[qctx][(l,n)][-1,h*D:(h+1)*D] for n in ('c_q','c_q2')};kk={n:cc[kctx][(l,n)][p,h*D:(h+1)*D] for n in ('c_k','c_k2')}
      for recipient in ('B','C'):
       for mode in ('native_value','third0','third1'):
        delta=None
        if mode!='native_value':
         token=third[int(mode[-1])][0];diff=token_bus(token)[h]-cc[recipient]['bus'][p,h].cuda();delta=(coef*F.linear(diff,attn.c_proj.weight[:,h*D:(h+1)*D])).cpu()
        jobs[(l,h,T,p)].append({'item':i,'variant':variant,'recipient':recipient,'query_context':qctx,'key_context':kctx,'coefficient':coef,'mode':mode,'ids':ee[recipient]['ids'],'query':qq,'key':kk,'delta':delta,'groups':[[before],[after]],'third':third})
 save()
 for (l,h,T,p),jj in sorted(jobs.items()):
  # Separate native-value and third jobs to keep additive hook semantics clear.
  for withdelta in (False,True):
   subset=[x for x in jj if (x['delta'] is not None)==withdelta]
   for begin in range(0,len(subset),8):
    batch=subset[begin:begin+8];qq={n:torch.stack([x['query'][n] for x in batch]) for n in ('c_q','c_q2')};kk={n:torch.stack([x['key'][n] for x in batch]) for n in ('c_k','c_k2')};delta=torch.stack([x['delta'] for x in batch]) if withdelta else None;logits=run(model,torch.tensor([x['ids'] for x in batch],device='cuda'),l,h,p,qq,kk,delta)
    for ix,x in enumerate(batch):
     margin=measure(logits[ix],x['groups'])['margin'];tm=measure(logits[ix],x['third'])['margin'];result['interventions'].append({k:x[k] for k in ('item','variant','recipient','query_context','key_context','coefficient','mode')}|{'layer':l,'head':h,'source_identity_margin':margin,'margin_delta':margin-native[(x['item'],x['variant'],x['recipient'])],'third_margin':tm})
  save()
 ix={(c['item'],c['variant'],c['recipient'],c['layer'],c['head'],c['query_context'],c['key_context'],c['mode']):c for c in result['interventions']}
 for c in result['interventions']:
  if c['mode']!='native_value':continue
  key=tuple(c[k] for k in ('item','variant','recipient','layer','head','query_context','key_context'));effect=ix[key+('third0',)]['third_margin']-ix[key+('third1',)]['third_margin'];fam='entity_binding' if c['recipient']=='B' else 'sequence_copy';pred=float(np.sign(locked[fam][f"{c['layer']}:{c['head']}"]['identity_map_sign']*c['coefficient']));result['third_comparisons'].append({k:c[k] for k in ('item','variant','recipient','layer','head','query_context','key_context','coefficient')}|{'third_identity_effect':effect,'predicted_sign':pred,'sign_correct':bool(np.sign(effect)==pred)})
 result['audits']={'self_qk_max_margin_delta':max(abs(c['margin_delta']) for c in result['interventions'] if c['mode']=='native_value' and c['query_context']==c['key_context']==c['recipient']),'matched_tokens_and_positions_asserted':True,'projection_hook_counts_asserted':True,'finite':all(np.isfinite(c['source_identity_margin']) and np.isfinite(c['third_margin']) for c in result['interventions']),'source_unchanged':cfg['source_sha256']==hashlib.sha256(src.read_bytes()).hexdigest(),'native_cache_unchanged':cfg['native_cache_sha256']==hashlib.sha256(cp.read_bytes()).hexdigest()};result['status']='complete';save();print('COMPLETE',flush=True)
if __name__=='__main__':main()

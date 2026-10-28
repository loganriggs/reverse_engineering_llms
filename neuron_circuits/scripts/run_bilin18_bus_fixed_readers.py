#!/usr/bin/env python3
"""H537: factorial freezing of downstream native QK and contextual V during bus swaps."""
import json,time,hashlib
from collections import defaultdict
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer
from run_bilin18_diverse_causal_tasks import ROOT,ART,groups,continuation_id,measure
from run_bilin18_qkov_causal_edges import load_full_model
from audit_bilin18_mlp_discriminators import TOKENIZER
STEM='bilin18_bus_fixed_readers'
torch.set_num_threads(6)
MODES={'free':(), 'fixed_qk':('c_q','c_k','c_q2','c_k2'), 'fixed_v':('c_v',), 'fixed_qkv':('c_q','c_k','c_q2','c_k2','c_v'), 'fixed_qkv_mlp':('c_q','c_k','c_q2','c_k2','c_v','mlp')}
@torch.inference_mode()
def run(model,ids,capture=False,injection=None,mode='free',frozen=None):
 d=model.config.n_embd;x0=F.rms_norm(model.embedding(ids),(d,));x=x0;bus=None;cache={};count=0
 for l,b in enumerate(model.blocks):
  handles=[]
  for name in ('c_q','c_k','c_q2','c_k2','c_v','mlp'):
   if capture or (injection is not None and l>injection[0] and name in MODES[mode]):
    def hook(m,args,out,name=name,l=l):
     nonlocal count
     if capture:cache[(l,name)]=out.cpu()
     if injection is not None and l>injection[0] and name in MODES[mode]:count+=1;return frozen[(l,name)].to(out)
     return out
    handles.append((b.mlp if name=='mlp' else getattr(b.attn,name)).register_forward_hook(hook))
  try:x,bus=b(x,bus,x0)
  finally:
   for h in handles:h.remove()
  if l==0 and capture:cache['bus']=bus.cpu()
  if injection is not None and l==injection[0]:bus=injection[1].to(bus)
 logits=30*torch.tanh(model.lm_head(F.rms_norm(x,(d,))[:,-1]).float()/30)
 if injection is not None:assert count==(17-injection[0])*len(MODES[mode])
 return logits,cache
@torch.inference_mode()
def main():
 start=time.time();source=ART/'bilin18_task_route_localization.json';prior=json.loads(source.read_text());mp=ART/'bilin18_diverse_causal_tasks_manifest.json';rows=[r for r in json.loads(mp.read_text()) if r['family'] in prior['selected_layers']]
 cfg={'experiment':'H537','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'manifest_sha256':hashlib.sha256(mp.read_bytes()).hexdigest(),'layers':{f:d['value_bus'] for f,d in prior['selected_layers'].items()},'modes':{k:list(v) for k,v in MODES.items()},'conditions':['base<-counterfactual','counterfactual<-base','base<-nuisance','base<-self'],'scope':'Freeze recipient-native projection outputs at every token in every layer strictly AFTER selected bus injection. Q/K pairs are frozen before their native RMS normalization and rotation, hence both routing factors are fixed. V is the local contextual value projection, not the persistent bus. MLP freezing substitutes the native entire output at all positions. No reselection or pair filtering. Equal lengths required for bus swaps.','hypotheses':['Fixed QK and V retain most bus effect if fixed-reader token-value transport suffices.','Loss under QK fixing implicates routing feedback conditional on intervention.','Additional loss under MLP fixing implicates downstream MLP response.'],'precision':'float32 compute after shared-loader bfloat16 weight rounding','panel':'Reused H533 discovery and transfer, separately reported; no new document or prompt.'}
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2)+'\n');print('LOCKED',flush=True)
 tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);model=load_full_model('cuda')
 for mod in (model.embedding,model.blocks,model.lm_head):
  mod.float().eval()
  for p in mod.parameters():p.requires_grad_(False)
 examples={};native={};cache={};lp={};buckets=defaultdict(list)
 for r in rows:
  for v,text in enumerate(r['prompts']+[r['nuisance']]):
   key=(r['id'],v);gs=[[continuation_id(tok,text,t) for t in g] for g in groups(r['family'],r['answers'])];assert all(all(t is not None for t in g) for g in gs);examples[key]={'ids':tok.encode(text,add_special_tokens=False),'groups':gs};buckets[len(examples[key]['ids'])].append(key)
 for length,kk in sorted(buckets.items()):
  for begin in range(0,len(kk),8):
   keys=kk[begin:begin+8];logits,cc=run(model,torch.tensor([examples[k]['ids'] for k in keys],device='cuda'),capture=True)
   for ix,key in enumerate(keys):cache[key]={n:v[ix].clone() for n,v in cc.items()};native[key]=measure(logits[ix],examples[key]['groups']);lp[key]=logits[ix].log_softmax(0).cpu()
 print('native cached',len(cache),flush=True)
 result={'config':cfg,'interventions':[],'skipped':[],'native':[dict(id=k[0],variant=k[1],**v) for k,v in native.items()],'status':'running'};out=ART/f'{STEM}.json'
 def save():result['elapsed_seconds']=time.time()-start;out.write_text(json.dumps(result,indent=2)+'\n')
 jobs=defaultdict(list)
 for r in rows:
  for rec,don in ((0,1),(1,0),(0,2),(0,0)):
   if len(examples[(r['id'],rec)]['ids'])!=len(examples[(r['id'],don)]['ids']):result['skipped'].append({'id':r['id'],'recipient':rec,'donor':don});continue
   for mode in MODES:jobs[(cfg['layers'][r['family']],mode,len(examples[(r['id'],rec)]['ids']))].append((r,rec,don))
 for (layer,mode,length),jj in sorted(jobs.items()):
  for begin in range(0,len(jj),8):
   batch=jj[begin:begin+8];keys=[(r['id'],rec) for r,rec,don in batch];ids=torch.tensor([examples[k]['ids'] for k in keys],device='cuda');bus=torch.stack([cache[(r['id'],don)]['bus'] for r,rec,don in batch]);frozen={(l,n):torch.stack([cache[k][(l,n)] for k in keys]) for l in range(layer+1,18) for n in MODES[mode]};logits,_=run(model,ids,injection=(layer,bus),mode=mode,frozen=frozen)
   for ix,(r,rec,don) in enumerate(batch):
    key=(r['id'],rec);mm=measure(logits[ix],examples[key]['groups']);delta=mm['margin']-native[key]['margin'];newlp=logits[ix].log_softmax(0).cpu();kl=float((lp[key].exp()*(lp[key]-newlp)).sum());result['interventions'].append(dict(id=r['id'],family=r['family'],split=r['split'],layer=layer,mode=mode,recipient=rec,donor=don,margin=mm['margin'],delta=delta,donor_directed_delta=delta*(1 if rec==1 else -1),KL=kl))
  save()
 print('interventions finished',len(result['interventions']),flush=True)
 old={(c['id'],c['recipient'],c['donor']):c for c in prior['interventions'] if c['component']=='value_bus' and c['layer']==cfg['layers'][c['family']]}
 result['audits']={'self_max_margin_delta':max(abs(c['delta']) for c in result['interventions'] if c['recipient']==c['donor']),'self_max_abs_KL':max(abs(c['KL']) for c in result['interventions'] if c['recipient']==c['donor']),'free_replay_h536_max_margin_error':max(abs(c['margin']-old[(c['id'],c['recipient'],c['donor'])]['margin']) for c in result['interventions'] if c['mode']=='free'),'freeze_hook_counts_asserted':True,'finite':all(np.isfinite(c['margin']) and np.isfinite(c['KL']) for c in result['interventions']),'manifest_unchanged':cfg['manifest_sha256']==hashlib.sha256(mp.read_bytes()).hexdigest(),'source_unchanged':cfg['source_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()};result['status']='complete';save();print('COMPLETE',flush=True)
if __name__=='__main__':main()

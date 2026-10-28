#!/usr/bin/env python3
"""H536: compare attention writes, final-position residuals and persistent value bus."""
import json,time,hashlib
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from run_bilin18_diverse_causal_tasks import ROOT,ART,groups,continuation_id,measure
from run_bilin18_qkov_causal_edges import load_full_model
from audit_bilin18_mlp_discriminators import TOKENIZER
STEM='bilin18_task_route_localization'
torch.set_num_threads(6)

@torch.inference_mode()
def run(model,ids,capture=False,patch=None):
 d=model.config.n_embd;x0=F.rms_norm(model.embedding(ids),(d,));x=x0;bus=None;cache={'attention':[],'residual':[]}
 for l,block in enumerate(model.blocks):
  h=None
  if capture or (patch is not None and patch[0]==l and patch[1] in ('attention','value_bus')):
   def hook(m,args,out):
    y,v=out
    if capture:cache['attention'].append(y[:,-1].cpu())
    if patch is not None and patch[0]==l:
     if patch[1]=='attention':y=y.clone();y[:,-1]=patch[2].to(y)
     elif patch[1]=='value_bus':v=patch[2].to(v)
    return y,v
   h=block.attn.register_forward_hook(hook)
  try:x,bus=block(x,bus,x0)
  finally:
   if h:h.remove()
  if patch is not None and patch[0]==l and patch[1]=='residual':x=x.clone();x[:,-1]=patch[2].to(x)
  if capture:
   cache['residual'].append(x[:,-1].cpu())
   if l==0:cache['value_bus']=bus.cpu()
 return 30*torch.tanh(model.lm_head(F.rms_norm(x,(d,))[:,-1]).float()/30),cache

@torch.inference_mode()
def main():
 start=time.time();src=ART/'bilin18_diverse_causal_tasks_manifest.json';rows=[r for r in json.loads(src.read_text()) if r['family'] in ('entity_binding','sequence_copy')];old=json.loads((ART/'bilin18_diverse_causal_tasks.json').read_text())
 cfg={'experiment':'H536','manifest_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'families':['entity_binding','sequence_copy'],'components':['attention','residual','value_bus'],'discovery_layers':list(range(18)),'layer_selection':'Max mean bidirectional donor-directed contrast over discovery pairs, separately per family/component. Residual excludes layer17 (trivial terminal readout control); value_bus excludes layer17 (no downstream consumers). Attention includes17. No individual pair selection.','transfer':'All transfer pairs at frozen component-specific layer.','scope':'Attention write or residual last position only; entire persistent value bus by aligned positions only when donor/recipient token lengths match. Other upstream states remain recipient-native. Bus injection occurs after the patched layer attention, influencing only later attention. The scopes differ and are not equal-size replacements.','controls':['both task directions','base-to-nuisance','self donor','terminal residual positive control','terminal bus negative control'],'hypotheses':['Attention or residual patches carry substantially more binding/copying contrast than previously selected MLP writes.','Correct task donors produce greater changes than nuisance donors.','Persistent raw-value bus is sufficient to transmit some task change.'],'precision':'float32 compute after bfloat16-rounded shared loader','fineweb_new_documents':0}
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2)+'\n');print('LOCKED',flush=True)
 tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);model=load_full_model('cuda')
 for module in (model.embedding,model.blocks,model.lm_head):
  module.float().eval()
  for p in module.parameters():p.requires_grad_(False)
 examples={};cache={};native={};buckets=defaultdict(list)
 for r in rows:
  for v,text in enumerate(r['prompts']+[r['nuisance']]):
   gs=[[continuation_id(tok,text,t) for t in g] for g in groups(r['family'],r['answers'])];assert all(all(t is not None for t in g) for g in gs)
   key=(r['id'],v);examples[key]={'ids':tok.encode(text,add_special_tokens=False),'groups':gs};buckets[len(examples[key]['ids'])].append(key)
 for length,keys in sorted(buckets.items()):
  for begin in range(0,len(keys),12):
   kk=keys[begin:begin+12];logits,c=run(model,torch.tensor([examples[k]['ids'] for k in kk],device='cuda'),capture=True)
   for ix,key in enumerate(kk):
    native[key]=measure(logits[ix],examples[key]['groups']);cache[key]={'attention':torch.stack([z[ix] for z in c['attention']]),'residual':torch.stack([z[ix] for z in c['residual']]),'value_bus':c['value_bus'][ix]}
 result={'config':cfg,'interventions':[],'skipped':[],'selected_layers':{},'status':'running'};out=ART/f'{STEM}.json'
 def save():result['elapsed_seconds']=time.time()-start;out.write_text(json.dumps(result,indent=2)+'\n')
 def execute(jobs):
  buckets=defaultdict(list)
  for r,l,typ,rec,don in jobs:
   if typ=='value_bus' and len(examples[(r['id'],rec)]['ids'])!=len(examples[(r['id'],don)]['ids']):
    result['skipped'].append({'id':r['id'],'layer':l,'component':typ,'recipient':rec,'donor':don,'reason':'unequal token lengths'});continue
   buckets[(l,typ,len(examples[(r['id'],rec)]['ids']))].append((r,rec,don))
  for (l,typ,length),jj in sorted(buckets.items()):
   for begin in range(0,len(jj),12):
    batch=jj[begin:begin+12];ee=[examples[(r['id'],rec)] for r,rec,don in batch];replacement=torch.stack([cache[(r['id'],don)][typ][l] if typ!='value_bus' else cache[(r['id'],don)][typ] for r,rec,don in batch]);logits,_=run(model,torch.tensor([e['ids'] for e in ee],device='cuda'),patch=(l,typ,replacement))
    for ix,(r,rec,don) in enumerate(batch):
     mm=measure(logits[ix],ee[ix]['groups']);delta=mm['margin']-native[(r['id'],rec)]['margin'];result['interventions'].append(dict(id=r['id'],family=r['family'],split=r['split'],layer=l,component=typ,recipient=rec,donor=don,margin=mm['margin'],delta=delta,donor_directed_delta=delta*(1 if rec==1 else -1)))
   save()
  print('batch phase complete',len(jobs),flush=True)
 jobs=[(r,l,typ,rec,don) for r in rows if r['split']=='discovery' for l in range(18) for typ in cfg['components'] for rec,don in ((0,1),(1,0),(0,2),(0,0))];execute(jobs)
 for fam in cfg['families']:
  result['selected_layers'][fam]={}
  for typ in cfg['components']:
   scores=[np.mean([c['donor_directed_delta'] for c in result['interventions'] if c['family']==fam and c['component']==typ and c['layer']==l and c['recipient']!=c['donor'] and c['donor']<2]) for l in range(18 if typ=='attention' else 17)]
   result['selected_layers'][fam][typ]=int(np.nanargmax(scores))
 save();jobs=[(r,result['selected_layers'][r['family']][typ],typ,rec,don) for r in rows if r['split']=='transfer' for typ in cfg['components'] for rec,don in ((0,1),(1,0),(0,2),(0,0))];execute(jobs)
 selferr=max(abs(c['delta']) for c in result['interventions'] if c['recipient']==c['donor']);terminalres=[];terminalbus=[]
 for c in result['interventions']:
  if c['layer']==17 and c['component']=='residual':terminalres.append(abs(c['margin']-native[(c['id'],c['donor'])]['margin']))
  if c['layer']==17 and c['component']=='value_bus':terminalbus.append(abs(c['delta']))
 result['audits']={'self_max_margin_delta':selferr,'terminal_residual_donor_margin_max_error':max(terminalres),'terminal_bus_max_margin_delta':max(terminalbus),'finite':all(np.isfinite(c['margin']) for c in result['interventions']),'manifest_unchanged':cfg['manifest_sha256']==hashlib.sha256(src.read_bytes()).hexdigest()};result['status']='complete';save();print('COMPLETE',flush=True)
if __name__=='__main__':main()

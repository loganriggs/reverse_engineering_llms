"""H843 causal task-state location: depth, token positions, persistent-state control."""
import json,time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_diverse_causal_tasks import ROOT,ART,forward,measure
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_early_local_geometry import sha
S='bilin18_task_state_location'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'data':ART/'bilin18_candidate_task_screen_captures.pt','screen':ART/'bilin18_candidate_task_screen.json','boundary':ART/'bilin18_candidate_boundary_controls.json','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};cfg={'experiment':'H843','hashes':hashes,'design':'All80 competent H840 task pairs bidirectional. Final-position block-state donor swaps at all18 layers. At layers1/2 also equal-length-only positional whole-sequence, token-differing positions, token-identical positions, and complete dynamic state (residual sequence + persistent L0 value bus + x0) donor swaps.','alignment':'Sequence intervention only exact equal token lengths; mask from indexwise token inequality. No semantic alignment claim. Unequal pairs retained for final-position sweep, excluded and listed only for sequence swaps.','controls':'Final-block swap must reproduce donor logits. Complete dynamic-state early swap must reproduce donor logits. Native runner agrees with established runner. Identity final-position swap at every depth on one batch.','scope':'Reused exploratory panel. Residual-only interventions retain recipient x0 and persistent value bus; complete-state arm changes both. Last-layer transfer is architectural positive control, not localized discovery.','next':'Use depth and position results to choose causal sites for representation validation; do not identify a feature solely by reconstruction or final-layer transfer.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');data=torch.load(paths['data'],weights_only=False);ex=data['examples'];ix={(e['id'],e['variant']):i for i,e in enumerate(ex)};screen=json.loads(paths['screen'].read_text());eligible={(r['family'],r['split']) for r in screen['competence'] if r['pass']};pairs=[e for e in ex if e['variant']==0 and (e['family'],e['split']) in eligible];model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 native={};caps={};actual=0;nativeerr=0;identerr=0;terminalerr=0;completeerr=0
 def run(ids,capture=False,patch=None):
  x0=F.rms_norm(model.embedding(ids),(1152,));x=x0;bus=None;c={}
  if capture:c['x0']=x0.cpu()
  for l,block in enumerate(model.blocks):
   x,bus=block(x,bus,x0)
   if capture:
    c[l,'last']=x[:,-1].cpu()
    if l in [1,2]:c[l,'sequence']=x.cpu();c[l,'bus']=bus.cpu()
   if patch is not None and l==patch['layer']:
    mode=patch['mode'];don=patch['donors']
    if mode=='last':x=x.clone();x[:,-1]=torch.stack([caps[j][l,'last'] for j in don]).to(x)
    else:
     repl=torch.stack([caps[j][l,'sequence'] for j in don]).to(x)
     if mode in ['sequence','complete']:x=repl
     else:x=torch.where(patch['mask'][:,:,None],repl,x)
     if mode=='complete':bus=torch.stack([caps[j][l,'bus'] for j in don]).to(bus);x0=torch.stack([caps[j]['x0'] for j in don]).to(x0)
  return 30*torch.tanh(model.lm_head(F.rms_norm(x,(1152,))[:,-1]).float()/30),c
 buckets=defaultdict(list)
 for e in pairs:
  for v in [0,1]:
   i=ix[e['id'],v];buckets[len(ex[i]['ids'])].append(i)
 first=True
 for length,inds in sorted(buckets.items()):
  for st in range(0,len(inds),16):
   batch=inds[st:st+16];ids=torch.tensor([ex[i]['ids'] for i in batch],device='cuda');lo,c=run(ids,True);actual+=len(batch)
   for k,i in enumerate(batch):native[i]=lo[k].double().cpu();caps[i]={key:value[k] for key,value in c.items()}
   if first:
    ref,_=forward(model,ids);actual+=len(batch);nativeerr=float((ref-lo).abs().max())
    for l in range(18):
     y,_=run(ids,patch={'layer':l,'mode':'last','donors':batch});actual+=len(batch);identerr=max(identerr,float((y-lo).abs().max()))
    first=False
 jobs=defaultdict(list);excluded=[]
 for e in pairs:
  equal=len(ex[ix[e['id'],0]]['ids'])==len(ex[ix[e['id'],1]]['ids'])
  if not equal:excluded.append(e['id'])
  for rec,don in [(0,1),(1,0)]:
   i,j=ix[e['id'],rec],ix[e['id'],don];base={'id':e['id'],'family':e['family'],'split':e['split'],'recipient':rec,'donor':don,'i':i,'j':j,'equal_length':equal}
   for l in range(18):jobs[l,'last',len(ex[i]['ids'])].append(base)
   if equal:
    for l in [1,2]:
     for mode in ['sequence','changed','unchanged','complete']:jobs[l,mode,len(ex[i]['ids'])].append(base)
 records=[]
 for (l,mode,length),jj in sorted(jobs.items()):
  for st in range(0,len(jj),24):
   batch=jj[st:st+24];ids=torch.tensor([ex[r['i']]['ids'] for r in batch],device='cuda');donors=[r['j'] for r in batch];patch={'layer':l,'mode':mode,'donors':donors}
   if mode in ['changed','unchanged']:
    mask=ids!=torch.tensor([ex[j]['ids'] for j in donors],device='cuda');patch['mask']=mask if mode=='changed' else ~mask
   y,_=run(ids,patch=patch);actual+=len(batch)
   for k,r in enumerate(batch):
    i,j=r['i'],r['j'];b=measure(native[i],ex[i]['groups'])['margin'];dm=measure(native[j],ex[j]['groups'])['margin'];value=measure(y[k],ex[i]['groups'])['margin'];sgn=1 if r['recipient']==1 else -1;lp=native[i].log_softmax(0);kl=float((lp.exp()*(lp-y[k].double().cpu().log_softmax(0))).sum());records.append({**r,'layer':l,'mode':mode,'donor_directed_delta':sgn*(value-b),'native_answer_gap':sgn*(dm-b),'KL':kl})
    if l==17 and mode=='last':terminalerr=max(terminalerr,float((y[k].double().cpu()-native[j]).abs().max()))
    if mode=='complete':completeerr=max(completeerr,float((y[k].double().cpu()-native[j]).abs().max()))
  print('DONE',l,mode,length,flush=True)
 summary=[]
 for l,mode in sorted({(r['layer'],r['mode']) for r in records}):
  for fam in sorted({e['family'] for e in pairs}):
   for subset in ['all','equal_length']:
    rr=[r for r in records if (r['layer'],r['mode'],r['family'])==(l,mode,fam) and (subset=='all' or r['equal_length'])];summary.append({'layer':l,'mode':mode,'family':fam,'subset':subset,'pairs':len(rr)//2,'delta':float(np.mean([r['donor_directed_delta'] for r in rr])),'gap_fraction':sum(r['donor_directed_delta'] for r in rr)/sum(r['native_answer_gap'] for r in rr)})
 old=json.loads(paths['boundary'].read_text());old={(r['layer'],r['id'],r['recipient']):r for r in old['records'] if r['kind']=='block'};repeaterr=max(abs(r['donor_directed_delta']-old[r['layer'],r['id'],r['recipient']]['donor_directed_delta']) for r in records if r['layer'] in [1,2] and r['mode']=='last');audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'native_logits_max_abs':nativeerr,'identity_logits_max_abs':identerr,'final_block_donor_logits_max_abs':terminalerr,'complete_state_donor_logits_max_abs':completeerr,'H842_repeat_max_margin_abs':repeaterr,'whole_model_forwards':actual,'test_evaluations':0,'finite':all(np.isfinite(r['KL']) for r in records)};assert audits['hashes_unchanged'] and max(nativeerr,identerr,terminalerr,completeerr,repeaterr)<1e-4 and audits['finite'];torch.save({'captures':caps,'native':native},ART/f'{S}_captures.pt');(ART/f'{S}.json').write_text(json.dumps({'config':cfg,'records':records,'summary':summary,'sequence_excluded':excluded,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(audits,flush=True)
if __name__=='__main__':main()

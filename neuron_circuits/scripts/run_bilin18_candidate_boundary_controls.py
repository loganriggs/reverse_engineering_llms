"""H842 distinguish weak early MLP task route from weak residual-state route."""
import json,time
from pathlib import Path
from collections import defaultdict
import torch
import numpy as np
from run_bilin18_diverse_causal_tasks import ROOT,ART,forward,measure
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_early_local_geometry import sha
S='bilin18_candidate_boundary_controls'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'captures':ART/'bilin18_candidate_task_screen_captures.pt','screen':ART/'bilin18_candidate_task_screen.json','swaps':ART/'bilin18_candidate_task_swaps.json','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};cfg={'experiment':'H842','hashes':hashes,'design':'Post-H841 diagnostic: same80 competent pairs, sites1/2, bidirectional full MLP-output donor and full block-output donor swaps at final position. Retain native recipient attention bus. This is a boundary-level intervention, not feature localization.','controls':'Identity replacement on every captured site/type; compare fullMLP-output effects against H841 full normalizedMLP-input donor swaps, which should implement same module computation.','scope':'Exploratory reused task panel; input-vs-write equivalence controls numerical/model correctness, whole block control tests bypassing early MLP.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');data=torch.load(paths['captures'],weights_only=False);ex=data['examples'];ix={(e['id'],e['variant']):i for i,e in enumerate(ex)};screen=json.loads(paths['screen'].read_text());eligible={(r['family'],r['split']) for r in screen['competence'] if r['pass']};pairs=[e for e in ex if e['variant']==0 and (e['family'],e['split']) in eligible];model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 native={};caps={};buckets=defaultdict(list);actual=0;ident=0.
 for e in pairs:
  for v in [0,1]:
   i=ix[e['id'],v];buckets[len(ex[i]['ids'])].append(i)
 for length,inds in sorted(buckets.items()):
  for st in range(0,len(inds),16):
   batch=inds[st:st+16];ids=torch.tensor([ex[i]['ids'] for i in batch],device='cuda');captured={};hooks=[]
   for l in [1,2]:
    for kind in ['mlp','block']:
     def capture(mod,args,out,l=l,kind=kind):captured[l,kind]=(out if kind=='mlp' else out[0])[:,-1].cpu()
     hooks.append((model.blocks[l].mlp if kind=='mlp' else model.blocks[l]).register_forward_hook(capture))
   try:logits,_=forward(model,ids)
   finally:
    for h in hooks:h.remove()
   actual+=len(batch)
   for k,i in enumerate(batch):
    native[i]=logits[k].double().cpu()
    for key,value in captured.items():caps[key,i]=value[k]
   for l in [1,2]:
    for kind in ['mlp','block']:
     def identity(mod,args,out):
      value=out if kind=='mlp' else out[0];value=value.clone();value[:,-1]=captured[l,kind].to(value);return value if kind=='mlp' else (value,out[1])
     h=(model.blocks[l].mlp if kind=='mlp' else model.blocks[l]).register_forward_hook(identity)
     try:yy,_=forward(model,ids)
     finally:h.remove()
     actual+=len(batch);ident=max(ident,float((yy-logits).abs().max()))
 records=[];jobs=defaultdict(list)
 for l in [1,2]:
  for kind in ['mlp','block']:
   for e in pairs:
    for rec,don in [(0,1),(1,0)]:
     i,j=ix[e['id'],rec],ix[e['id'],don];jobs[l,kind,len(ex[i]['ids'])].append((e,rec,don,i,j))
 for (l,kind,length),jj in sorted(jobs.items()):
  for st in range(0,len(jj),24):
   batch=jj[st:st+24];ids=torch.tensor([ex[i]['ids'] for e,rec,don,i,j in batch],device='cuda');replacement=torch.stack([caps[(l,kind),j] for e,rec,don,i,j in batch]).cuda()
   def patch(mod,args,out):
    val=out if kind=='mlp' else out[0];val=val.clone();val[:,-1]=replacement;return val if kind=='mlp' else (val,out[1])
   h=(model.blocks[l].mlp if kind=='mlp' else model.blocks[l]).register_forward_hook(patch)
   try:y,_=forward(model,ids)
   finally:h.remove()
   actual+=len(batch)
   for k,(e,rec,don,i,j) in enumerate(batch):
    b=measure(native[i],ex[i]['groups'])['margin'];v=measure(y[k],ex[i]['groups'])['margin'];dm=measure(native[j],ex[j]['groups'])['margin'];sgn=1 if rec==1 else -1;lp=native[i].log_softmax(0);kl=float((lp.exp()*(lp-y[k].double().cpu().log_softmax(0))).sum());records.append({'layer':l,'kind':kind,'family':e['family'],'split':e['split'],'id':e['id'],'recipient':rec,'donor':don,'donor_directed_delta':sgn*(v-b),'native_answer_gap':sgn*(dm-b),'KL':kl})
  print('DONE',l,kind,length,flush=True)
 old=json.loads(paths['swaps'].read_text());old={(r['layer'],r['id'],r['recipient']):r for r in old['records'] if r['arm']=='full_input'};equiv=max(abs(r['donor_directed_delta']-old[r['layer'],r['id'],r['recipient']]['donor_directed_delta']) for r in records if r['kind']=='mlp');summ=[]
 for l in [1,2]:
  for kind in ['mlp','block']:
   for fam in sorted({e['family'] for e in pairs}):
    rr=[r for r in records if (r['layer'],r['kind'],r['family'])==(l,kind,fam)];summ.append({'layer':l,'kind':kind,'family':fam,'delta':float(np.mean([r['donor_directed_delta'] for r in rr])),'gap_fraction':sum(r['donor_directed_delta'] for r in rr)/sum(r['native_answer_gap'] for r in rr),'mean_KL':float(np.mean([r['KL'] for r in rr]))})
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'identity_logits_max_abs':ident,'input_vs_output_swap_max_margin_difference':equiv,'whole_model_forwards':actual,'test_evaluations':0,'all_finite':all(np.isfinite(r['KL']) for r in records)};assert audits['hashes_unchanged'] and ident<1e-4 and equiv<1e-4 and audits['all_finite'];torch.save({'captures':caps,'native':native},ART/f'{S}_captures.pt');(ART/f'{S}.json').write_text(json.dumps({'config':cfg,'records':records,'summary':summ,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(json.dumps({'summary':summ,'audits':audits}),flush=True)
if __name__=='__main__':main()

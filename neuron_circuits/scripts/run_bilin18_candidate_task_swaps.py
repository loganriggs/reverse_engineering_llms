"""H841 frozen candidate causal transplantation at normalized MLP inputs."""
import json,time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from run_bilin18_diverse_causal_tasks import ROOT,ART,forward,measure
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_early_local_geometry import sha
S='bilin18_candidate_task_swaps'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest')
 paths={'screen':ART/'bilin18_candidate_task_screen.json','captures':ART/'bilin18_candidate_task_screen_captures.pt','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()}
 cfg={'experiment':'H841','hashes':hashes,'design':'All16 frozen H839 candidates, both seeds, all80 pairs in H840 competent cells. No individual-success selection. Replace decoded component at final normalized MLP input using native input + donor_component - recipient_component. Preserve residual stream and all other positions. No renormalization after patch. Controlled module input, not natural residual-state swap.','arms':'Task swaps bidirectional, each with paired plus/minus ambient Gaussian direction at exactly same L2 dose. Same-answer nuisance donor to endpoint0, component swap. Full normalized-input task swap as positive control, both directions. Identity native input audit per layer/batch shape.','random':'One frozen Gaussian unit per layer/kind/candidate, common across seeds and pairs; both signs. Limited directional null, no universal random-subspace comparison.','metrics':'Donor-directed answer margin change; KL(native||patched); task donor minus recipient native gap; token/position matching and dose. Zero vectors assigned exact identity outcome without extra forward.','scope':'Reused exploratory task suite, no fresh confirmation. All failures/inactive components retained; no semantic labels inferred.'}
 (ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');screen=json.loads(paths['screen'].read_text());data=torch.load(paths['captures'],weights_only=False);ex=data['examples'];ix={(e['id'],e['variant']):i for i,e in enumerate(ex)};eligible={(c['family'],c['split']) for c in screen['competence'] if c['pass']};pairs=[e for e in ex if e['variant']==0 and (e['family'],e['split']) in eligible];model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 native={};actual=0;auditcap=0.;identityerr=0.;buckets=defaultdict(list)
 for e in pairs:
  for v in [0,1,2]:
   i=ix[e['id'],v];buckets[len(ex[i]['ids'])].append(i)
 for length,inds in sorted(buckets.items()):
  for st in range(0,len(inds),16):
   batch=inds[st:st+16];ids=torch.tensor([ex[i]['ids'] for i in batch],device='cuda');lo,_=forward(model,ids);actual+=len(batch)
   for k,i in enumerate(batch):native[i]=lo[k].double().cpu()
   for l in [1,2]:
    def identity(mod,args):
     nonlocal auditcap
     u=args[0];ref=data['normalized_premlp'][l][batch].to(u);auditcap=max(auditcap,float((u[:,-1]-ref).abs().max()));u=u.clone();u[:,-1]=ref;return (u,)
    h=model.blocks[l].mlp.register_forward_pre_hook(identity)
    try:y,_=forward(model,ids)
    finally:h.remove()
    actual+=len(batch);identityerr=max(identityerr,float((y-lo).abs().max()))
 assert auditcap<1e-4 and identityerr<1e-4
 jobs=[]
 for (l,kind),comp in data['components'].items():
  for ci in range(4):
   gen=torch.Generator().manual_seed(841000+l*100+ci+(10 if kind=='block8' else 0));q=torch.randn(1152,generator=gen);q=q/q.norm()
   for seed in [8351,8352]:
    z=comp['decoded'][seed][:,ci]
    for e in pairs:
     for rec,don in [(0,1),(1,0),(0,2)]:
      i,j=ix[e['id'],rec],ix[e['id'],don];delta=z[j]-z[i];mode='task' if don!=2 else 'nuisance';base={'layer':l,'kind':kind,'candidate':ci,'seed':seed,'id':e['id'],'family':e['family'],'split':e['split'],'recipient':rec,'donor':don,'i':i,'j':j,'mode':mode,'dose':float(delta.norm()),'same_final_token':ex[i]['ids'][-1]==ex[j]['ids'][-1],'same_length':len(ex[i]['ids'])==len(ex[j]['ids'])}
      jobs.append((dict(base,arm='component'),delta))
      if mode=='task':
       for sign in [-1,1]:jobs.append((dict(base,arm='random',random_sign=sign),sign*q*delta.norm()))
 for l in [1,2]:
  for e in pairs:
   for rec,don in [(0,1),(1,0)]:
    i,j=ix[e['id'],rec],ix[e['id'],don];delta=data['normalized_premlp'][l][j]-data['normalized_premlp'][l][i];jobs.append(({'layer':l,'kind':'full_input','candidate':-1,'seed':0,'id':e['id'],'family':e['family'],'split':e['split'],'recipient':rec,'donor':don,'i':i,'j':j,'mode':'task','dose':float(delta.norm()),'same_final_token':ex[i]['ids'][-1]==ex[j]['ids'][-1],'same_length':len(ex[i]['ids'])==len(ex[j]['ids']),'arm':'full_input'},delta))
 records=[];bucket=defaultdict(list);zero=0
 def add(job,logits):
  r=job.copy();i,j=r['i'],r['j'];base=measure(native[i],ex[i]['groups'])['margin'];value=measure(logits,ex[i]['groups'])['margin'];don=measure(native[j],ex[j]['groups'])['margin'];sgn=1 if r['recipient']==1 else -1;lp=native[i].log_softmax(0);kl=float((lp.exp()*(lp-logits.double().cpu().log_softmax(0))).sum());r.update(delta_margin=value-base,donor_directed_delta=sgn*(value-base),native_answer_gap=sgn*(don-base),KL=kl);records.append(r)
 for job,delta in jobs:
  if job['dose']==0:add(job,native[job['i']]);zero+=1
  else:bucket[job['layer'],len(ex[job['i']]['ids'])].append((job,delta))
 for (l,length),jj in sorted(bucket.items()):
  for st in range(0,len(jj),24):
   batch=jj[st:st+24];ids=torch.tensor([ex[j['i']]['ids'] for j,d in batch],device='cuda');delta=torch.stack([d for j,d in batch]).cuda()
   def patch(mod,args):
    u=args[0].clone();u[:,-1]+=delta.to(u);return (u,)
   h=model.blocks[l].mlp.register_forward_pre_hook(patch)
   try:logits,_=forward(model,ids)
   finally:h.remove()
   actual+=len(batch)
   for k,(j,d) in enumerate(batch):add(j,logits[k])
  print('DONE',l,length,len(jj),'records',len(records),'/',len(jobs),flush=True)
  (ART/f'{S}_partial.json').write_text(json.dumps({'records':records,'planned':len(jobs)}))
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'native_capture_max_abs':auditcap,'identity_logits_max_abs':identityerr,'whole_model_forwards':actual,'logical_interventions':len(jobs),'exact_zero_skipped':zero,'test_evaluations':0,'all_finite':all(np.isfinite(r['KL']) and np.isfinite(r['delta_margin']) for r in records),'minimum_KL':min(r['KL'] for r in records)};assert audits['hashes_unchanged'] and audits['all_finite'] and len(records)==len(jobs)
 (ART/f'{S}.json').write_text(json.dumps({'config':cfg,'records':records,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(audits,flush=True)
if __name__=='__main__':main()

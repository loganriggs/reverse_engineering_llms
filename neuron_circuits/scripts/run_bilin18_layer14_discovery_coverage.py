"""H661: single exact attention-layer repairs and leave-one-predicted controls."""
import json,time,hashlib
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn.functional as F
import numpy as np
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_reader_operation_format import run
STEM='bilin18_layer14_discovery_coverage'
@torch.no_grad()
def main():
 torch.set_num_threads(6);start=time.time();paths={'manifest':ART/'bilin18_operation_paraphrase_panel_manifest.json','captures':ART/'bilin18_operation_paraphrase_panel_captures.pt','means':ART/'bilin18_boundary_attention_wording_locked.pt','exact':ART/'bilin18_finite_factorial_captures.pt','previous':ART/'bilin18_attention_layer_repairs_captures.pt','runner':Path(__file__)};hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()};rows=json.loads(paths['manifest'].read_text());cap=torch.load(paths['captures'],weights_only=False);means=torch.load(paths['means'],weights_only=False)['new'];exact=torch.load(paths['exact'],weights_only=False);source,target=exact['source'],exact['target'];schemes=['ordered4','alternating4','leave_one_out7','exact14']
 cfg=dict(experiment='H662',hashes=hashes,design='96 existing transfer cases: change only layer14 attention prediction; keep new-wording boundary and other attention means fixed, also test exact boundary. Ordered first4 mean, alternating4 item offsets1/3/5/7 modulo8, all7 nonself items, exact14 control. Exclude evaluated item from fits. Original eval items4-7 unchanged. No fitting to outputs.',limitations='Retrospective cross-validation: other previously evaluated items enter alternative training sets, never the evaluated item. Alternating balance is by item index, not all semantic properties. No prospective validation or claim more data always helps.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:
  mod.float().eval()
  for p in mod.parameters():p.requires_grad_(False)
 buckets=defaultdict(list)
 for i,s in enumerate(source):buckets[len(rows[s]['ids'])].append(i)
 basea=torch.empty(96,8,1152);attnevals=0
 for l in range(10,18):
  block=model.blocks[l]
  for ii in buckets.values():
   for st in range(0,len(ii),16):
    bb=ii[st:st+16];ss=[source[i] for i in bb];x=torch.stack([cap['caches'][l][i] for i in ss]).cuda();x0=torch.stack([cap['x0'][i] for i in ss]).cuda();bus=torch.stack([cap['bus'][i] for i in ss]).cuda();a,_=block.attn(F.rms_norm(block.lambdas[0]*x+block.lambdas[1]*x0,(1152,)),bus);basea[bb,l-10]=a[:,-1].cpu();attnevals+=len(bb)
 lookup={(r['family'],r['control'],r['item'],r['form']):i for i,r in enumerate(rows)};allindices=[i for i,r in enumerate(rows) if r['control'] in ['copy','task']];all14={};ab=defaultdict(list)
 for i in allindices:ab[(rows[i]['item']<4,len(rows[i]['ids']))].append(i)
 block=model.blocks[14]
 for ii in ab.values():
  for st in range(0,len(ii),16):
   bb=ii[st:st+16];x=torch.stack([cap['caches'][14][i] for i in bb]).cuda();x0=torch.stack([cap['x0'][i] for i in bb]).cuda();bus=torch.stack([cap['bus'][i] for i in bb]).cuda();aa,_=block.attn(F.rms_norm(block.lambdas[0]*x+block.lambdas[1]*x0,(1152,)),bus);attnevals+=len(bb)
   for k,i in enumerate(bb):all14[i]=aa[k,-1].cpu().double()
 alternatives={key:[] for key in ['alternating4','leave_one_out7']};fit_records=[];geometry=[];meanreplay=0.
 for i,sr in enumerate(source):
  row=rows[sr];f,form,item=row['family'],row['form'],row['item'];ds=torch.stack([all14[lookup[f,'task',it,form]]-all14[lookup[f,'copy',it,form]] for it in range(8)]);truth=ds[item];meanreplay=max(meanreplay,float((ds[:4].mean(0)-means[f,form]['attention'][4]).abs().max()))
  for scheme in ['ordered4','alternating4','leave_one_out7']:
   train=list(range(4)) if scheme=='ordered4' else [(item+o)%8 for o in [1,3,5,7]] if scheme=='alternating4' else [it for it in range(8) if it!=item];assert item not in train and len(train)==len(set(train));pred=ds[train].mean(0);fit_records.append(dict(id=row['id'],scheme=scheme,excluded_item=item,training_items=train));geometry.append(dict(id=row['id'],family=f,scheme=scheme,squared_error=float((pred-truth).square().sum()),target_squared=float(truth.square().sum())))
   if scheme!='ordered4':alternatives[scheme].append(pred)
 alternatives={k:torch.stack(v) for k,v in alternatives.items()};print('MEAN_REPLAY',meanreplay,flush=True);assert meanreplay<1e-5
 predb=cap['native_captures'][9][source].double()+torch.stack([means[rows[s]['family'],rows[s]['form']]['boundary'] for s in source]);preda=basea.double()+torch.stack([means[rows[s]['family'],rows[s]['form']]['attention'] for s in source]);donorb=cap['native_captures'][9][target];donora=torch.stack([exact['donor_attention'][l] for l in range(10,18)],1);forwards=0
 def evaluate(boundary,scheme):
  nonlocal forwards
  logits=torch.empty(96,model.lm_head.weight.shape[0])
  for ii in buckets.values():
   for st in range(0,len(ii),16):
    bb=ii[st:st+16];hooks=[]
    def bpatch(mod,args,out):
     y,bus=out;y=y.clone();y[:,-1]=(donorb if boundary=='exact' else predb)[bb].to(y);return y,bus
    hooks.append(model.blocks[9].register_forward_hook(bpatch))
    for l in range(10,18):
     def apatch(mod,args,out,l=l):
      y,bus=out;y=y.clone();v=preda[bb,l-10]
      if l==14:
       if scheme=='exact14':v=donora[bb,l-10]
       elif scheme!='ordered4':v=basea[bb,l-10]+alternatives[scheme][bb]
      y[:,-1]=v.to(y);return y,bus
     hooks.append(model.blocks[l].attn.register_forward_hook(apatch))
    try:z,_=run(model,torch.tensor([rows[source[i]]['ids'] for i in bb],device='cuda'));logits[bb]=z.cpu()
    finally:
     for h in hooks:h.remove()
    forwards+=len(bb)
  return logits
 tasklog=exact['donor_logits'].double().log_softmax(-1);p=tasklog.exp();records=[];outputs={}
 for b in ['predicted','exact']:
  for scheme in schemes:
   z=evaluate(b,scheme);outputs[b,scheme]=z;lp=z.double().log_softmax(-1);kl=(p*(tasklog-lp)).sum(-1)
   for i,s in enumerate(source):
    r=rows[s];t=rows[target[i]];ids=t['candidate_ids'];aid=t['answer_id'];records.append(dict(id=r['id'],family=r['family'],form=r['form'],item=r['item'],boundary=b,scheme=scheme,task_kl=float(kl[i]),target_ce=float(-lp[i,aid]),candidate_correct=ids[int(z[i,ids].argmax())]==aid,top1_correct=int(z[i].argmax())==aid))
  print('COMPLETE',b,flush=True)
 prior=torch.load(paths['previous'],weights_only=False)['logits'];replays={str((b,scheme)):float((outputs[b,scheme]-prior[b,0 if scheme=='ordered4' else 16]).abs().max()) for b in ['predicted','exact'] for scheme in ['ordered4','exact14']};assert max(replays.values())==0;audits=dict(hashes_unchanged={k:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[k] for k,p in paths.items()},whole_model_forwards=forwards,cached_attention_evaluations=attnevals,ordered_mean_replay_max_abs=meanreplay,evaluated_item_excluded=True,previous_baseline_replay_max_abs=replays,finite=all(np.isfinite(r['task_kl']) for r in records));torch.save(dict(logits=outputs,source=source,target=target),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,fit_records=fit_records,geometry=geometry,audits=audits,elapsed_seconds=time.time()-start),indent=2));print('AUDITS',audits,flush=True)
if __name__=='__main__':main()

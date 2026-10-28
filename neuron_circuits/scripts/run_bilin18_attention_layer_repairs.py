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
STEM='bilin18_attention_layer_repairs'
@torch.no_grad()
def main():
 torch.set_num_threads(6);start=time.time();paths={'manifest':ART/'bilin18_operation_paraphrase_panel_manifest.json','captures':ART/'bilin18_operation_paraphrase_panel_captures.pt','means':ART/'bilin18_boundary_attention_wording_locked.pt','exact':ART/'bilin18_finite_factorial_captures.pt','previous':ART/'bilin18_boundary_attention_wording_captures.pt','runner':Path(__file__)};hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()};rows=json.loads(paths['manifest'].read_text());cap=torch.load(paths['captures'],weights_only=False);means=torch.load(paths['means'],weights_only=False)['new'];exact=torch.load(paths['exact'],weights_only=False);source,target=exact['source'],exact['target'];masks=[0,255]+[1<<k for k in range(8)]+[255^(1<<k) for k in range(8)];assert len(set(masks))==18
 cfg=dict(experiment='H661',hashes=hashes,design='96 heldout transfer copy prompts, frozen new-wording discovery means. For each predicted/exact boundary state: all attention predicted, all exact,8 single exact repairs,8 leave-one-predicted conditions. Each mask bit chooses exact donor attention output at layer10+bit; other writes use receiver-native baseline plus new discovery mean. MLPs run naturally; no refit or layer selection.',limitations='Exact donor repairs localize error, not deployable prediction or causal shares. Layer ranking exploratory; no best-layer validation. All6 families retained; known task/form and native receiver baseline.')
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
 predb=cap['native_captures'][9][source].double()+torch.stack([means[rows[s]['family'],rows[s]['form']]['boundary'] for s in source]);preda=basea.double()+torch.stack([means[rows[s]['family'],rows[s]['form']]['attention'] for s in source]);donorb=cap['native_captures'][9][target];donora=torch.stack([exact['donor_attention'][l] for l in range(10,18)],1);forwards=0
 def evaluate(boundary,mask):
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
      y,bus=out;y=y.clone();y[:,-1]=(donora if mask&(1<<(l-10)) else preda)[bb,l-10].to(y);return y,bus
     hooks.append(model.blocks[l].attn.register_forward_hook(apatch))
    try:z,_=run(model,torch.tensor([rows[source[i]]['ids'] for i in bb],device='cuda'));logits[bb]=z.cpu()
    finally:
     for h in hooks:h.remove()
    forwards+=len(bb)
  return logits
 tasklog=exact['donor_logits'].double().log_softmax(-1);p=tasklog.exp();records=[];outputs={}
 for b in ['predicted','exact']:
  for mask in masks:
   z=evaluate(b,mask);outputs[b,mask]=z;lp=z.double().log_softmax(-1);kl=(p*(tasklog-lp)).sum(-1)
   for i,s in enumerate(source):
    r=rows[s];t=rows[target[i]];ids=t['candidate_ids'];aid=t['answer_id'];records.append(dict(id=r['id'],family=r['family'],form=r['form'],item=r['item'],boundary=b,mask=mask,task_kl=float(kl[i]),target_ce=float(-lp[i,aid]),candidate_correct=ids[int(z[i,ids].argmax())]==aid,top1_correct=int(z[i].argmax())==aid))
  print('COMPLETE',b,flush=True)
 prior=torch.load(paths['previous'],weights_only=False)['logits'];replays={str((b,m)):float((outputs[b,m]-prior['new' if b=='predicted' else 'exact','new' if m==0 else 'exact']).abs().max()) for b in ['predicted','exact'] for m in [0,255]};assert max(replays.values())==0;audits=dict(hashes_unchanged={k:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[k] for k,p in paths.items()},whole_model_forwards=forwards,cached_attention_evaluations=attnevals,previous_baseline_replay_max_abs=replays,finite=all(np.isfinite(r['task_kl']) for r in records));torch.save(dict(logits=outputs,source=source,target=target),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,audits=audits,elapsed_seconds=time.time()-start),indent=2));print('AUDITS',audits,flush=True)
if __name__=='__main__':main()

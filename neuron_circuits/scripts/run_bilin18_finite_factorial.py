"""H658: exact upstream/attention/MLP factorial for finite output recovery."""
import json,time,hashlib
from pathlib import Path
from collections import defaultdict
import torch
import numpy as np
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_reader_operation_format import run
STEM='bilin18_finite_factorial'
@torch.no_grad()
def main():
 torch.set_num_threads(6);start=time.time();paths={'manifest':ART/'bilin18_operation_paraphrase_panel_manifest.json','captures':ART/'bilin18_operation_paraphrase_panel_captures.pt','indices':ART/'bilin18_operation_switch_effects_locked.pt','previous':ART/'bilin18_finite_operation_edits.json','runner':Path(__file__)};hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()};allrows=json.loads(paths['manifest'].read_text());cap=torch.load(paths['captures'],weights_only=False);lock=torch.load(paths['indices'],weights_only=False);source,target=lock['source'],lock['target'];rows=[allrows[i] for i in source];assert len(rows)==96
 assert all(allrows[s]['ids'][-1]==allrows[t]['ids'][-1] for s,t in zip(source,target));assert all(torch.equal(cap['x0'][s][-1],cap['x0'][t][-1]) for s,t in zip(source,target))
 cfg=dict(experiment='H658',hashes=hashes,design='Exact full2x2x2 finite factorial on96 heldout copy prompts. Bit1 donor final-position block9 residual; bit2 donor attention output final position every layer10-17; bit4 donor normalized MLP input final position every layer10-17. Same input and suffix, donor second task wording. All masks retained. Full combination recovery check; native final-token x0 equal asserted.',limitations='Exact heldout donor states; localization controls not predictors or compact surrogate. Hybrid states need not be naturally realizable. KL, CE and correctness separate; factorial improvements are not causal shares.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:
  mod.float().eval()
  for p in mod.parameters():p.requires_grad_(False)
 attn={l:torch.empty(96,1152) for l in range(10,18)};forwards=0
 def evaluate(indices,mask=0,capture=False):
  nonlocal forwards
  buckets=defaultdict(list)
  for i,s in enumerate(indices):buckets[len(allrows[s]['ids'])].append(i)
  logits=torch.empty(96,model.lm_head.weight.shape[0])
  for ii in buckets.values():
   for st in range(0,len(ii),16):
    bb=ii[st:st+16];hooks=[]
    if mask&1:
     def upstream(mod,args,out):
      y,b=out;y=y.clone();y[:,-1]=cap['native_captures'][9][[target[i] for i in bb]].to(y);return y,b
     hooks.append(model.blocks[9].register_forward_hook(upstream))
    for l in range(10,18):
     if capture:
      def save(mod,args,out,l=l):attn[l][bb]=out[0][:,-1].cpu()
      hooks.append(model.blocks[l].attn.register_forward_hook(save))
     if mask&2:
      def editatt(mod,args,out,l=l):
       y,b=out;y=y.clone();y[:,-1]=attn[l][bb].to(y);return y,b
      hooks.append(model.blocks[l].attn.register_forward_hook(editatt))
     if mask&4:
      def editmlp(mod,args,l=l):
       y=args[0].clone();y[:,-1]=cap['normalized_mlp_contexts'][l][[target[i] for i in bb]].to(y);return (y,)+args[1:]
      hooks.append(model.blocks[l].mlp.register_forward_pre_hook(editmlp))
    try:z,_=run(model,torch.tensor([allrows[indices[i]]['ids'] for i in bb],device='cuda'));logits[bb]=z.cpu()
    finally:
     for h in hooks:h.remove()
    forwards+=len(bb)
  return logits
 donor=evaluate(target,capture=True);donor_replay=float((donor-cap['native_logits'][target]).abs().max());assert donor_replay<1e-3
 tasklog=donor.double().log_softmax(-1);p=tasklog.exp();records=[];outputs={}
 for mask in range(8):
  z=evaluate(source,mask);outputs[mask]=z;lp=z.double().log_softmax(-1);kl=(p*(tasklog-lp)).sum(-1)
  for i,r in enumerate(rows):
   t=allrows[target[i]];cs=t['candidate_ids'];aid=t['answer_id'];records.append(dict(id=r['id'],family=r['family'],form=r['form'],item=r['item'],mask=mask,task_kl=float(kl[i]),target_ce=float(-lp[i,aid]),candidate_correct=cs[int(z[i,cs].argmax())]==aid,top1_correct=int(z[i].argmax())==aid))
  print('COMPLETE',mask,flush=True)
 native_replay=float((outputs[0]-cap['native_logits'][source]).abs().max());joint_error=float((outputs[7]-donor).abs().max());joint_kl=max(r['task_kl'] for r in records if r['mask']==7);assert native_replay<1e-3 and joint_error<1e-3 and joint_kl<1e-8
 old=json.loads(paths['previous'].read_text());oldlook={r['id']:r for r in old['records'] if r['mode']=='exact_clamp' and r['alpha']==1};replay=max(abs(r['task_kl']-oldlook[r['id']]['task_kl']) for r in records if r['mask']==4);assert replay<1e-5
 audits=dict(hashes_unchanged={k:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[k] for k,p in paths.items()},whole_model_forwards=forwards,donor_native_replay_max_abs=donor_replay,native_replay_max_abs=native_replay,joint_recovery_logit_max_abs=joint_error,joint_recovery_kl_max=joint_kl,prior_mlp_only_kl_max_abs=replay,final_x0_equal=True,finite=all(np.isfinite(r['task_kl']) and np.isfinite(r['target_ce']) for r in records));torch.save(dict(logits=outputs,donor_logits=donor,donor_attention=attn,source=source,target=target),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,audits=audits,elapsed_seconds=time.time()-start),indent=2));print('AUDITS',audits,flush=True)
if __name__=='__main__':main()

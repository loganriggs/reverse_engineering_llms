"""H657: finite MLP-input edits, KL fidelity and task loss kept distinct."""
import json,time,hashlib
from pathlib import Path
from collections import defaultdict
import torch
import numpy as np
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_reader_operation_format import run
STEM='bilin18_finite_operation_edits'
@torch.no_grad()
def main():
 torch.set_num_threads(6);start=time.time();paths={'manifest':ART/'bilin18_operation_paraphrase_panel_manifest.json','captures':ART/'bilin18_operation_paraphrase_panel_captures.pt','means':ART/'bilin18_operation_switch_effects_locked.pt','runner':Path(__file__)};hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()};allrows=json.loads(paths['manifest'].read_text());cap=torch.load(paths['captures'],weights_only=False);locked=torch.load(paths['means'],weights_only=False);means=locked['means'];source=locked['source'];target=locked['target'];rows=[allrows[i] for i in source];families=sorted({r['family'] for r in rows});modes=['cross_wording','own','other_tasks','normmatched_other','constant','exact_delta','exact_clamp'];amplitudes=[.25,.5,1.]
 cfg=dict(experiment='H657',hashes=hashes,design='96 heldout copy prompts; final-position normalized MLP input edits at layers10-17;7 modes x fixed amplitudes0.25/0.5/1 plus native. Add frozen shifts to actual evolving MLP inputs, without renormalizing. Exact-delta uses donor-native minus copy-native contexts; exact-clamp interpolates actual evolving context toward donor-native context. Compare KL(task||edited), target CE, candidate and exact top1 separately. All original item splits retained.',limitations='Finite activation edits differ from derivative operator replacement. Exact controls use heldout donor activations; means use discovery only. No amplitude selection or behavioral optimization. Natural task prompt can be incorrect. Edit does not replace earlier processing or attention.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));shift={m:[] for m in modes};donor=torch.stack([torch.stack([cap['normalized_mlp_contexts'][l][t].double() for l in range(10,18)]) for t in target])
 for i,r in enumerate(rows):
  fam,form=r['family'],r['form']
  for m in modes:
   if m in ['exact_delta','exact_clamp']:v=torch.cat([cap['normalized_mlp_contexts'][l][target[i]].double()-cap['normalized_mlp_contexts'][l][source[i]].double() for l in range(10,18)])
   elif m in ['other_tasks','normmatched_other']:
    v=torch.stack([means[f,'task','original',form] for f in families if f!=fam]).mean(0)
    if m=='normmatched_other':v=v*means[fam,'task','original',form].norm()/v.norm()
   else:v=means[fam,'constant' if m=='constant' else 'task','new' if m=='own' else 'original',form]
   shift[m].append(v.reshape(8,1152))
 shift={m:torch.stack(v) for m,v in shift.items()};model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:
  mod.float().eval()
  for p in mod.parameters():p.requires_grad_(False)
 buckets=defaultdict(list)
 for i,r in enumerate(rows):buckets[len(r['ids'])].append(i)
 forwards=0
 def evaluate(mode=None,alpha=0.):
  nonlocal forwards
  logits=torch.empty(len(rows),model.lm_head.weight.shape[0])
  for ii in buckets.values():
   for st in range(0,len(ii),16):
    bb=ii[st:st+16];hooks=[]
    if mode is not None:
     for l in range(10,18):
      def edit(mod,args,l=l):
       y=args[0].clone();last=y[:,-1]
       if mode=='exact_clamp':y[:,-1]=last+alpha*(donor[bb,l-10].to(last)-last)
       else:y[:,-1]=last+alpha*shift[mode][bb,l-10].to(last)
       return (y,)+args[1:]
      hooks.append(model.blocks[l].mlp.register_forward_pre_hook(edit))
    try:z,_=run(model,torch.tensor([rows[i]['ids'] for i in bb],device='cuda'));logits[bb]=z.cpu()
    finally:
     for h in hooks:h.remove()
    forwards+=len(bb)
  return logits
 tasklog=cap['native_logits'][target].double().log_softmax(-1);taskp=tasklog.exp();copylog=cap['native_logits'][source].double().log_softmax(-1);copyp=copylog.exp();records=[];outputs={}
 def score(mode,alpha,z):
  lp=z.double().log_softmax(-1);kl=(taskp*(tasklog-lp)).sum(-1);copykl=(copyp*(copylog-lp)).sum(-1)
  for i,r in enumerate(rows):
   t=allrows[target[i]];ids=t['candidate_ids'];answer=t['answer_id'];records.append(dict(id=r['id'],family=r['family'],form=r['form'],item=r['item'],mode=mode,alpha=alpha,task_kl=float(kl[i]),copy_kl=float(copykl[i]),target_ce=float(-lp[i,answer]),candidate_correct=ids[int(z[i,ids].argmax())]==answer,top1_correct=int(z[i].argmax())==answer))
 native=evaluate();replay=float((native-cap['native_logits'][source]).abs().max());print('NATIVE_REPLAY',replay,flush=True);assert replay<1e-3;score('native',0.,native);score('task_reference',1.,cap['native_logits'][target]);zero=evaluate('exact_clamp',0.);zeroerror=float((zero-native).abs().max());assert zeroerror==0
 for m in modes:
  for alpha in amplitudes:
   z=evaluate(m,alpha);score(m,alpha,z);outputs[m,alpha]=z;print('COMPLETE',m,alpha,flush=True)
 audits=dict(hashes_unchanged={k:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[k] for k,p in paths.items()},whole_model_forwards=forwards,native_replay_max_abs=replay,zero_hook_max_abs=zeroerror,finite=all(np.isfinite(r['task_kl']) and np.isfinite(r['target_ce']) for r in records),prompt_count=len(rows));torch.save(dict(logits=outputs,source=source,target=target),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,audits=audits,elapsed_seconds=time.time()-start),indent=2));print('AUDITS',audits,flush=True)
if __name__=='__main__':main()

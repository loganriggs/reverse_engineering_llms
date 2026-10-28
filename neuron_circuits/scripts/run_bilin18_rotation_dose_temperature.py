"""H710: dose response and edited-distribution temperature projection."""
import torch,json,time,hashlib
from pathlib import Path
from collections import defaultdict
from transformers import AutoTokenizer
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_reader_operation_format import run
STEM='bilin18_rotation_dose_temperature'
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time();paths={'data':ART/'bilin18_early_bsf_data_captures.pt','swaps':ART/'bilin18_rotation_context_swap_captures.pt','old_results':ART/'bilin18_rotation_context_swap.json','model':SNAPSHOT/'pytorch_model.bin','runner':Path(__file__)};hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()};data=torch.load(paths['data'],weights_only=False);old=torch.load(paths['swaps'],weights_only=False);oldresults=json.loads(paths['old_results'].read_text());indices=[i for i,p in enumerate(old['pairs']) if p['direction'] in [121,208]];cfg=dict(experiment='H710',hashes=hashes,doses=[0.,.1,.25,.5,1.],design='All16directed H708pairs for posthoc-selected121/208, coordinate/PCA/random edits at0/0.1/0.25/0.5/1dose. Retain editedlogits. Fit inverse temperature beta in[0.05,20] minimizing KL(Pedited||softmax(beta*base_logits)) by convex moment-matching bisection, no donor used in fit. Evaluate donorward gain and observednexttoken CE separately.',limitations='Same selected validation cases, not independent confirmation. Temperature is one-parameter descriptive control, not a causal mediator or additive decomposition. Natural nexttoken CE ononeobserved continuation is not task competence.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 buckets=defaultdict(list)
 for i in indices:buckets[old['pairs'][i]['position']].append(i)
 meta=[];logitrows=[];forwards=0
 for mode in ['coordinate','pca','random']:
  for dose in cfg['doses']:
   for pos,inds in buckets.items():
    patch=torch.stack([old['patches'][i][mode]*dose for i in inds])
    def edit(m,args,out):
     x,bus=out;x=x.clone();x[:,-1]+=patch.to(x);return x,bus
    hook=model.blocks[1].register_forward_hook(edit)
    try:logits,_=run(model,torch.stack([data['all_token_ids'][2048+old['pairs'][i]['receiver']//16,:pos+1] for i in inds]).cuda())
    finally:hook.remove()
    for n,i in enumerate(inds):meta.append(dict(pair_index=i,**old['pairs'][i],mode=mode,dose=dose));logitrows.append(logits[n].cpu())
    forwards+=len(inds)
  print('DONE',mode,flush=True)
 del model;torch.cuda.empty_cache();edited_logits=torch.stack(logitrows);base=torch.stack([old['native_logits'][r['receiver']] for r in meta]).cuda().double();don=torch.stack([old['native_logits'][r['donor']] for r in meta]).cuda().double().log_softmax(-1);edit=edited_logits.cuda().double().log_softmax(-1);bp=base.log_softmax(-1);ep=edit.exp();dp=don.exp();centered=base-base.max(-1,keepdim=True).values;target=(ep*centered).sum(-1);lo=torch.full_like(target,.05);hi=torch.full_like(target,20.)
 for _ in range(50):
  mid=(lo+hi)/2;prob=(centered*mid[:,None]).softmax(-1);moment=(prob*centered).sum(-1);lo=torch.where(moment<target,mid,lo);hi=torch.where(moment>=target,mid,hi)
 beta=(lo+hi)/2;temp=(centered*beta[:,None]).log_softmax(-1);edit_to_base=(ep*(edit-bp)).sum(-1);edit_to_temp=(ep*(edit-temp)).sum(-1);assert float((edit_to_temp-edit_to_base).max())<1e-7
 baseline=(dp*(don-bp)).sum(-1);gain=baseline-(dp*(don-edit)).sum(-1);tgain=baseline-(dp*(don-temp)).sum(-1);tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);records=[];zeroerrors=[];replay=[]
 for k,r in enumerate(meta):
  doc=2048+r['receiver']//16;nextid=int(data['all_token_ids'][doc,r['position']+1]);top=ep[k].topk(5);rec=dict(**r,donor_kl_gain=float(gain[k]),temperature_donor_kl_gain=float(tgain[k]),gain_minus_temperature=float(gain[k]-tgain[k]),beta=float(beta[k]),temperature_boundary=bool(beta[k]<.05001 or beta[k]>19.999),native_entropy=float(-(bp[k].exp()*bp[k]).sum()),edited_entropy=float(-(ep[k]*edit[k]).sum()),edit_to_base_kl=float(edit_to_base[k]),edit_to_temperature_kl=float(edit_to_temp[k]),observed_next_token=tok.decode([nextid]),observed_next_token_ce_change=float(bp[k,nextid]-edit[k,nextid]),edited_top5=[dict(token=tok.decode([int(t)]),probability=float(p)) for p,t in zip(top.values,top.indices)]);records.append(rec)
  if r['dose']==0:zeroerrors.append(float(edit_to_base[k]))
  if r['dose']==1:
   original=next(v for v in oldresults['records'] if v['direction']==r['direction'] and v['receiver']==r['receiver'] and v['donor']==r['donor'] and v['mode']==r['mode']);replay.append(abs(rec['donor_kl_gain']-original['donor_kl_gain']))
 audits=dict(hashes_unchanged={k:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[k] for k,p in paths.items()},finite=all(torch.isfinite(torch.tensor(r['donor_kl_gain'])).item() for r in records),zero_dose_max_kl=max(zeroerrors),unit_dose_gain_replay_max_abs=max(replay),temperature_fit_never_worse_than_beta1=True,whole_model_forwards=forwards,test_evaluations=0);assert all(audits['hashes_unchanged'].values()) and audits['finite'] and max(zeroerrors)<1e-7 and max(replay)<1e-4;torch.save(dict(metadata=meta,edited_logits=edited_logits),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,audits=audits,elapsed_seconds=time.time()-start),indent=2));print(audits,flush=True)
if __name__=='__main__':main()

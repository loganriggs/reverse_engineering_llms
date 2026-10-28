"""H736: frozen representations on H732 fresh five-family panel."""
import json,time
from pathlib import Path
from collections import defaultdict
import torch,numpy as np
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_reader_operation_format import run
from run_bilin18_coactivation_group_swaps import sha
from run_bilin18_scalar_capacity_train import SignedTopK
from run_bilin18_learned_group_memberships import select_pairs,union_pairs
STEM='bilin18_scalar_capacity_function'
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time();paths={'fits':ART/'bilin18_verified_head_write_pca_fits.pt','models':ART/'bilin18_scalar_capacity_train_captures.pt','previous':ART/'bilin18_fresh_featurizer_transfer.json','fresh':ART/'bilin18_fresh_head_tasks_captures.pt','fresh_results':ART/'bilin18_fresh_head_tasks.json','manifest':ART/'bilin18_fresh_head_tasks_manifest.json','model':SNAPSHOT/'pytorch_model.bin','runner':Path(__file__),'sparse':ROOT/'scripts/run_bilin18_scalar_capacity_train.py','gates':ROOT/'scripts/run_bilin18_learned_group_memberships.py','forward':ROOT/'scripts/run_bilin18_reader_operation_format.py','loader':ROOT/'scripts/run_bilin18_qkov_causal_edges.py'};hashes={k:sha(p) for k,p in paths.items()};fits=torch.load(paths['fits'],weights_only=False)['fits'];states=torch.load(paths['models'],weights_only=False)['states'];prev=json.loads(paths['previous'].read_text());old=torch.load(paths['fresh'],weights_only=False);prior=json.loads(paths['fresh_results'].read_text());examples=old['examples'];rows=json.loads(paths['manifest'].read_text());sites={'agreement':('agreement_head',11,3),'capital_recall':('capital_head',13,0)}
 cfg=dict(experiment='H736',hashes=hashes,design='FrozenH735 separatelytrained scalarTopK8/16/32/64, all3seeds atL11H3/L13H0. SameH732all231contexts and5families. Taskedits F(donor)-F(receiver); nativeheadreplacement KL andintendedanswer-setNLL. Samewidth128/33024parameters, varyingactivecapacity. No functionalmodelselection; K8replayagainstH733.',interpretation='Retainallstrata, labelH732competencepass. Reportabsolute RMS effecterror universally; retention ratio only when abs(meanexactheadtaskgain)>=.25, threshold fixedbeforeevaluation. Nearzerohead effects are specificitycontrols. Failedcompetence cells diagnostic only.',limitations='FiniteH732panel; freshrelative torepresentation training butheadnative results alreadyinspected. No learnedgroup selection onthispanel. Doesnotestablish individualcomponentreuse. Intendedanswer-setNLL notsameasobservedtokenCE; reservednaturaltestuntouched.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 variants=[('identity',None,None)]+[('scalar_topk',seed,k) for k in [8,16,32,64] for seed in [7241,7242,7243]];keys=list(examples);records=[];fidelity=[];cache={};forwards=0;replay=[]
 previous={(r['site'],r['id'],r['receiver'],r['donor']):r for r in prior['records']};native=old['native_logits']
 for site,(native_site,layer,head) in sites.items():
  fit=fits[site];mu=fit['mean'].cuda().float();E=fit['write_pca']['encode'].cuda().float();B=fit['write_pca']['decode'].cuda().float();raw=torch.stack([old['head_coordinates'][k][native_site] for k in keys]).cuda();p=(raw-mu)@E
  for kind,seed,ns in variants:
   if kind=='identity':rec=raw.clone()
   else:
    state=states[site,ns,seed];sparse=SignedTopK(ns).cuda();sparse.load_state_dict(state['model']);rp=sparse(p/state['input_scale'])*state['input_scale'];rec=mu+rp@B
   lookup={k:rec[n].cpu() for n,k in enumerate(keys)};cache[site,kind,seed,ns]=lookup
   for mode in ['task','native_reconstruction']:
    jobs=defaultdict(list)
    for row in rows:
     if (row['id'],0) not in examples:continue
     for rv,dv in ([(0,1),(1,0),(0,2)] if mode=='task' else [(0,0),(1,1),(2,2)]):jobs[len(examples[row['id'],rv]['ids'])].append(dict(id=row['id'],family=row['family'],split=row['split'],receiver=rv,donor=dv,site=site,kind=kind,seed=seed,null_seed=ns,active_k=ns))
    for length,jj in jobs.items():
     for lo in range(0,len(jj),12):
      bb=jj[lo:lo+12];values=torch.stack([lookup[r['id'],r['donor']]-lookup[r['id'],r['receiver']] if mode=='task' else lookup[r['id'],r['receiver']] for r in bb]).cuda()
      def edit(m,args):
       y=args[0].clone()
       if mode=='task':y[:,-1,head*128:(head+1)*128]+=values.to(y)
       else:y[:,-1,head*128:(head+1)*128]=values.to(y)
       return (y,)
      hook=model.blocks[layer].attn.c_proj.register_forward_pre_hook(edit)
      try:logits,_=run(model,torch.tensor([examples[r['id'],r['receiver']]['ids'] for r in bb],device='cuda'))
      finally:hook.remove()
      forwards+=len(bb)
      for n,r in enumerate(bb):
       k=(r['id'],r['receiver']);gs=examples[k]['groups'];base=measure(native[k],gs);ed=measure(logits[n],gs);change=ed['margin']-base['margin']
       if mode=='task':
        sign=1 if r['donor'] in [0,2] else -1;records.append(dict(**r,margin_change=change,donor_directed_margin_change=sign*change,edited_donor_correct=ed['top1'][0 if r['donor']==2 else r['donor']]))
        if kind=='identity':replay.append(abs(change-previous[native_site,r['id'],r['receiver'],r['donor']]['margin_change']))
       else:
        bp=native[k].double().log_softmax(-1);ep=logits[n].cpu().double().log_softmax(-1);target=0 if r['receiver']==2 else r['receiver'];fidelity.append(dict(**r,native_to_reconstruction_kl=float((bp.exp()*(bp-ep)).sum()),intended_answer_set_nll_change=float(torch.logsumexp(bp[gs[target]],0)-torch.logsumexp(ep[gs[target]],0)),native_top1_correct=base['top1'][target],reconstructed_top1_correct=ed['top1'][target]))
   print('DONE',site,kind,seed,ns,flush=True)
 scalar_previous={(r['site'],r['id'],r['receiver'],r['donor'],r['seed']):r['margin_change'] for r in prev['task_records'] if r['kind']=='scalar_top8'};scalar_replay=max(abs(r['margin_change']-scalar_previous[r['site'],r['id'],r['receiver'],r['donor'],r['seed']]) for r in records if r['kind']=='scalar_topk' and r['active_k']==8);zero=max(r['native_to_reconstruction_kl'] for r in fidelity if r['kind']=='identity');audits=dict(hashes_unchanged={k:sha(p)==hashes[k] for k,p in paths.items()},finite=all(np.isfinite(r['native_to_reconstruction_kl']) for r in fidelity) and all(np.isfinite(r['margin_change']) for r in records),k8_h733_task_replay_max_abs=scalar_replay,identity_task_replay_max_abs=max(replay),identity_native_max_kl=zero,whole_model_forwards=forwards,test_evaluations=0);print(audits,flush=True);assert all(audits['hashes_unchanged'].values()) and audits['finite'] and max(replay)<1e-4 and zero<1e-7 and scalar_replay<1e-4;torch.save(dict(reconstructions=cache),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,competence=prior['summary'],task_records=records,fidelity=fidelity,audits=audits,elapsed_seconds=time.time()-start),indent=2))
if __name__=='__main__':main()

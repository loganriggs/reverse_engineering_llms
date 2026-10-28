"""H737: frozen representations on H732 fresh five-family panel."""
import json,time
from pathlib import Path
from collections import defaultdict
import torch,numpy as np
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_reader_operation_format import run
from run_bilin18_coactivation_group_swaps import sha

from run_bilin18_learned_group_memberships import select_pairs,union_pairs
STEM='bilin18_random_coordinate_capacity'
def orthogonal_basis(seed,dimension=128):
 gen=torch.Generator().manual_seed(seed);q,r=torch.linalg.qr(torch.randn(dimension,dimension,generator=gen,dtype=torch.double));return q*torch.sign(r.diag())[None,:]
def reconstruct_coordinates(p,Q,k):
 z=p@Q;mask=torch.zeros_like(z).scatter_(1,z.abs().topk(k,dim=-1).indices,1);return (z*mask)@Q.T,mask
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time();paths={'fits':ART/'bilin18_verified_head_write_pca_fits.pt','data':ART/'bilin18_verified_head_pca_data.pt','previous':ART/'bilin18_fresh_featurizer_transfer.json','fresh':ART/'bilin18_fresh_head_tasks_captures.pt','fresh_results':ART/'bilin18_fresh_head_tasks.json','manifest':ART/'bilin18_fresh_head_tasks_manifest.json','model':SNAPSHOT/'pytorch_model.bin','runner':Path(__file__),'sparse':ROOT/'scripts/run_bilin18_scalar_capacity_train.py','gates':ROOT/'scripts/run_bilin18_learned_group_memberships.py','forward':ROOT/'scripts/run_bilin18_reader_operation_format.py','loader':ROOT/'scripts/run_bilin18_qkov_causal_edges.py'};hashes={k:sha(p) for k,p in paths.items()};fits=torch.load(paths['fits'],weights_only=False)['fits'];data=torch.load(paths['data'],weights_only=False);prev=json.loads(paths['previous'].read_text());old=torch.load(paths['fresh'],weights_only=False);prior=json.loads(paths['fresh_results'].read_text());examples=old['examples'];rows=json.loads(paths['manifest'].read_text());sites={'agreement':('agreement_head',11,3),'capital_recall':('capital_head',13,0)}
 cfg=dict(experiment='H737',hashes=hashes,design='Samewrite-PCAcoordinates, signedadaptiveTopK8/16/32/64 using eitherPCAaxes or3fixedrandomorthogonalbases (QRofseededGaussianwithdiagonal-sign correction). Randomrotation inunwhitenedwritecoordinates, distinctfromH724whitenedrotation. Samebasis acrossK; allseeds, nofitting orselection. Full4096naturalvalidationvectors andH732freshtask/nativeKLassay. CompareexistingH735/H736learnedmodels separately; equalactivebudget/width notparameter-matchedtraining.',interpretation='Retainallstrata, labelH732competencepass. Reportabsolute RMS effecterror universally; retention ratio only when abs(meanexactheadtaskgain)>=.25, threshold fixedbeforeevaluation. Nearzerohead effects are specificitycontrols. Failedcompetence cells diagnostic only.',limitations='FiniteH732panel; freshrelative torepresentation training butheadnative results alreadyinspected. No learnedgroup selection onthispanel. Doesnotestablish individualcomponentreuse. Intendedanswer-setNLL notsameasobservedtokenCE; reservednaturaltestuntouched.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 variants=[('identity',None,None)]+[('adaptive_pca',None,k) for k in [8,16,32,64]]+[('random_orthogonal',seed,k) for k in [8,16,32,64] for seed in [7371,7372,7373]];keys=list(examples);records=[];fidelity=[];cache={};forwards=0;replay=[];validation=[];bases={};orth_errors=[];full_errors=[];budgets=[]
 previous={(r['site'],r['id'],r['receiver'],r['donor']):r for r in prior['records']};native=old['native_logits']
 for site,(native_site,layer,head) in sites.items():
  fit=fits[site];mu=fit['mean'].cuda().float();E=fit['write_pca']['encode'].cuda().float();B=fit['write_pca']['decode'].cuda().float();raw=torch.stack([old['head_coordinates'][k][native_site] for k in keys]).cuda();p=(raw-mu)@E;vp=(data['activations'][site][2048:].reshape(-1,128).cuda()-mu)@E
  for kind,seed,ns in variants:
   if kind=='identity':rec=raw.clone()
   else:
    Q=torch.eye(128,device='cuda') if kind=='adaptive_pca' else orthogonal_basis(seed).cuda().float();bases[site,kind,seed]=Q.cpu();orth_errors.append(float((Q.T@Q-torch.eye(128,device='cuda')).abs().max()));rp,mask=reconstruct_coordinates(p,Q,ns);rec=mu+rp@B;vr,vmask=reconstruct_coordinates(vp,Q,ns);validation.append(dict(site=site,kind=kind,seed=seed,k=ns,write_mse=float((vr-vp).square().sum()/vp.square().sum())));budgets.append(bool((mask.sum(-1)==ns).all()) and bool((vmask.sum(-1)==ns).all()));full,_=reconstruct_coordinates(p,Q,128);full_errors.append(float((full-p).norm()/p.norm()))
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
 zero=max(r['native_to_reconstruction_kl'] for r in fidelity if r['kind']=='identity');audits=dict(hashes_unchanged={k:sha(p)==hashes[k] for k,p in paths.items()},finite=all(np.isfinite(r['native_to_reconstruction_kl']) for r in fidelity) and all(np.isfinite(r['margin_change']) for r in records),orthogonality_max_abs=max(orth_errors),full_coordinate_roundtrip_max_relative=max(full_errors),active_budgets_correct=all(budgets),identity_task_replay_max_abs=max(replay),identity_native_max_kl=zero,whole_model_forwards=forwards,test_evaluations=0);print(audits,flush=True);assert all(audits['hashes_unchanged'].values()) and audits['finite'] and max(replay)<1e-4 and zero<1e-7 and max(orth_errors)<1e-5 and max(full_errors)<1e-5 and all(budgets);torch.save(dict(reconstructions=cache,bases=bases),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,competence=prior['summary'],natural_validation=validation,task_records=records,fidelity=fidelity,audits=audits,elapsed_seconds=time.time()-start),indent=2))
if __name__=='__main__':main()

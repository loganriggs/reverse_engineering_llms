"""H728: exact reconstruction-aware pair selection on fixed memberships.
Uses input activation for inference only; no validation parameter fitting.
"""
import json,time,hashlib
from pathlib import Path
import torch,numpy as np
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_coactivation_group_swaps import sha
from run_bilin18_verified_head_sparse import SignedBSF
from run_bilin18_frozen_dictionary_groups import greedy_union
STEM='bilin18_reconstruction_group_gates'
@torch.no_grad()
def best_support(x,z,D,bias,supports,chunk=512):
 # Enumerate exact fixed-coefficient reconstructions, including decoder cross terms.
 best=torch.full((len(x),),torch.inf,device=x.device,dtype=x.dtype);idx=torch.zeros(len(x),device=x.device,dtype=torch.long)
 for start in range(0,len(supports),chunk):
  pred=(z[:,None,:]*supports[None,start:start+chunk,:])@D+bias
  err=(pred-x[:,None,:]).square().sum(-1);v,i=err.min(-1);update=v<best;best=torch.where(update,v,best);idx=torch.where(update,i+start,idx)
 return supports[idx],best
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time();paths={'data':ART/'bilin18_verified_head_pca_data.pt','fits':ART/'bilin18_verified_head_write_pca_fits.pt','models':ART/'bilin18_verified_head_sparse_captures.pt','memberships':ART/'bilin18_frozen_dictionary_groups_captures.pt','sparse':ROOT/'scripts/run_bilin18_verified_head_sparse.py','gates':ROOT/'scripts/run_bilin18_frozen_dictionary_groups.py','runner':Path(__file__)};hashes={k:sha(p) for k,p in paths.items()};data=torch.load(paths['data'],weights_only=False);fits=torch.load(paths['fits'],weights_only=False)['fits'];states=torch.load(paths['models'],weights_only=False)['states'];members=torch.load(paths['memberships'],weights_only=False)['memberships'];docs=sorted(range(2048,2304),key=lambda d:hashlib.sha256(f'H722:{d}'.encode()).hexdigest())[:32]
 cfg=dict(experiment='H728',hashes=hashes,documents=docs,design='All54 frozenH726membershipsets and6scalar dictionaries. Same32 hash-selected natural validationdocs usedbyH725/H727, all16 sampledpositions=512vectors/head. Compare coefficientenergygreedy vs globallybest distinctgrouppair byexact reconstructionerror, holding coefficients/decoder/memberships fixed. Enumerate all8128pairs (496disjoint), unioncounts sharedatomsonce. Also scalarTop8 andscalarTopK with K equal exactpairuniqueoccupancy perexample. No training/fitting or tasklabels. Inputactivation is available at inference; exact search computationallyexpensive.',limitations='Limitedreusedvalidationpanel; inputreconstruction-optimal doesnot imply outputfaithful. Fixedcoefficients only, not leastsquares refitting or optimal learned memberships. Same2groupbudget but occupancy maychange withgate. Originalreservedtestuntouched.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));records=[];recons={};dominance=[];replays=[];relative_dominance=[];relative_replay=[]
 for site,fit in fits.items():
  p=(data['activations'][site][docs].reshape(-1,128).cuda()-fit['mean'].cuda().float())@fit['write_pca']['encode'].cuda().float()
  for seed in [7241,7242,7243]:
   st=states[site,'bsf1',seed];model=SignedBSF(1).cuda();model.load_state_dict(st['model']);x=p/st['input_scale'];z=model.encoder(x);D=model.decoder;bias=model.bias;den=x.square().sum();base=(model(x)-x).square().sum()/den;order=z.abs().argsort(-1,descending=True);ranks=order.argsort(-1)
   for (ss,sd,kind,ns),M in members.items():
    if (ss,sd)!=(site,seed):continue
    M=M.cuda();ij=torch.triu_indices(len(M),len(M),offset=1,device='cuda');supports=torch.maximum(M[ij[0]],M[ij[1]]);qs=[];err=[]
    for lo in range(0,len(x),16):
     q,e=best_support(x[lo:lo+16],z[lo:lo+16],D,bias,supports);qs.append(q);err.append(e)
    q=torch.cat(qs);e=torch.cat(err);energyq,_=greedy_union(z,M);ep=(z*energyq)@D+bias;exact=(z*q)@D+bias;direct=(exact-x).square().sum(-1);dominance.append(float((e-(ep-x).square().sum(-1)).max()));replays.append(float((e-direct).abs().max()));double_exact=((z.double()*q.double())@D.double()+bias.double()-x.double()).square().sum(-1);double_energy=((z.double()*energyq.double())@D.double()+bias.double()-x.double()).square().sum(-1);norm=x.double().square().sum(-1).clamp_min(1e-12);relative_dominance.append(float(((double_exact-double_energy)/norm).max()));relative_replay.append(float(((e.double()-double_exact).abs()/norm).max()));k=q.sum(-1);kmask=(ranks<k[:,None]).to(z);kpred=(z*kmask)@D+bias
    rec=dict(site=site,seed=seed,kind=kind,null_seed=ns,scalar_top8_mse=float(base),energy_mse=float((ep-x).square().sum()/den),exact_pair_mse=float(e.sum()/den),occupancy_matched_scalar_mse=float((kpred-x).square().sum()/den),exact_unique_atoms=float(k.mean()),energy_unique_atoms=float(energyq.sum(-1).mean()),same_support_fraction=float((q==energyq).all(-1).float().mean()),candidate_pairs=len(supports));records.append(rec);recons[site,seed,kind,ns]=dict(exact_mask=q.cpu(),exact_reconstruction=(exact*st['input_scale']).cpu());print(site,seed,kind,ns,rec['energy_mse'],rec['exact_pair_mse'],flush=True)
 audits=dict(hashes_unchanged={k:sha(p)==hashes[k] for k,p in paths.items()},max_exact_minus_energy_sse=max(dominance),max_scoring_replay_error=max(replays),max_double_precision_relative_dominance_error=max(relative_dominance),max_double_precision_relative_score_error=max(relative_replay),finite=all(np.isfinite(r['exact_pair_mse']) for r in records),whole_model_forwards=0,test_evaluations=0);print(audits,flush=True);assert all(audits['hashes_unchanged'].values()) and audits['finite'] and max(relative_dominance)<1e-6 and max(relative_replay)<1e-6
 torch.save(dict(documents=docs,reconstructions=recons),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,audits=audits,elapsed_seconds=time.time()-start),indent=2));print(audits,flush=True)
if __name__=='__main__':main()

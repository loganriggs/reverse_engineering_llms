"""H757: discrete anchored membership learning for captured coordinate energy."""
import json,time
from pathlib import Path
import torch,numpy as np
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_coactivation_group_swaps import sha
from run_bilin18_verified_head_sparse import SignedBSF
from run_bilin18_frozen_dictionary_groups import rewire
STEM='bilin18_l5_shared_group_memberships'
def select_pairs(w,M,rounds=4):
 a=(w@M.T).argmax(-1);scores=(w*(1-M[a]))@M.T;scores.scatter_(1,a[:,None],-torch.inf);b=scores.argmax(-1);pairs=torch.stack([a,b],-1)
 for _ in range(rounds):
  for slot in [0,1]:
   other=pairs[:,1-slot];scores=(w*(1-M[other]))@M.T;scores.scatter_(1,other[:,None],-torch.inf);pairs[:,slot]=scores.argmax(-1)
 return pairs

def union_pairs(M,pairs):return torch.maximum(M[pairs[:,0]],M[pairs[:,1]])

def update_memberships(w,M,pairs):
 # Coordinate ascent with fixed assignments: other group's atoms already count once.
 M=M.clone()
 for g in range(len(M)):
  rows,slots=(pairs==g).nonzero(as_tuple=True)
  if len(rows)==0:continue
  other=pairs[rows,1-slots];benefit=(w[rows]*(1-M[other])).sum(0);benefit[g]=-torch.inf;neighbors=benefit.topk(3).indices;M[g].zero_();M[g,g]=1;M[g,neighbors]=1
 return M
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time();paths={'data':ART/'bilin18_l5_natural_featurizer_baselines_data.pt','fits':ART/'bilin18_l5_natural_featurizer_baselines_fits.pt','models':ART/'bilin18_l5_sparse_featurizers_captures.pt','sparse':ROOT/'scripts/run_bilin18_verified_head_sparse.py','rewiring':ROOT/'scripts/run_bilin18_frozen_dictionary_groups.py','runner':Path(__file__)};hashes={k:sha(p) for k,p in paths.items()};data=torch.load(paths['data'],weights_only=False);fits=torch.load(paths['fits'],weights_only=False)['fits'];models=torch.load(paths['models'],weights_only=False)['states']
 cfg=dict(experiment='H757',hashes=hashes,design='Frozen six H755 scalar dictionaries. Learn128 groups of4 shared atoms with own anchor, fixed512 memberships. Two selected groups decode exact union (<=8 unique atoms), never double count. Initializations: stable-half Top8-support correlation and random anchored groups. Ten discrete membership-update rounds maximizing training raw squared coefficient energy; assignment greedy plus4 alternating bestresponses; retain best training score including initial. Evaluate both initial/learned models and three degree-preserving rewires per learned fit. Same validation gate. Scalar8 and per-example union-size-matched scalar controls. No task data in fitting.',limitations='Proxy energy objective, nonorthogonal decoder, frozen dictionary and coefficients, local optimization with anchors. Not jointly trained shared-gate architecture. Fixed group/membership counts substitute for membership penalty. Randomized memberships preserve row/column degrees but need not retain anchors. Nominal8slots, actual unique count recorded. Validation evaluated only; reused split and no original reserved-test evaluation.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));records=[];trajectories=[];saved={};membership_deltas=[];union_errors=[];degree_errors=[]
 for site,fit in fits.items():
  p=(data['activations'][site].reshape(-1,128).cuda()-fit['mean'].cuda().float())@fit['write_pca']['encode'].cuda().float()
  for seed in [7551,7552,7553]:
   st=models[site,1,seed];model=SignedBSF(1).cuda();model.load_state_dict(st['model']);z=model.encoder(p/st['input_scale']);w=z[:32768].square();vz=z[32768:];val=p[32768:];den=val.square().sum();variants={};gen=torch.Generator(device='cuda').manual_seed(seed+730000);rand=torch.rand(128,128,device='cuda',generator=gen);rand.fill_diagonal_(-1);randomM=torch.eye(128,device='cuda').scatter_(1,rand.topk(3,dim=1).indices,1)
   _,support=model.encode(p[:32768]/st['input_scale']);corrs=[]
   for half in [support[:16384],support[16384:]]:
    y=half-half.mean(0);energy=y.square().sum(0);corrs.append(y.T@y/(energy[:,None]*energy[None,:]).sqrt().clamp_min(1e-12))
   corr=torch.minimum(*corrs);corr.fill_diagonal_(-2);rawM=torch.eye(128,device='cuda').scatter_(1,corr.topk(3,dim=1).indices,1)
   for init,M0 in [('raw',rawM),('random',randomM)]:
    M=M0.clone();best=-1.;bestM=M.clone();trajectory=[]
    for epoch in range(11):
     pairs=select_pairs(w,M);q=union_pairs(M,pairs);score=float((w*q).sum()/w.sum());trajectory.append(dict(epoch=epoch,train_captured_energy_fraction=score))
     if score>best:best=score;bestM=M.clone()
     if epoch==10:break
     newM=update_memberships(w,M,pairs);newscore=float((w*union_pairs(newM,pairs)).sum()/w.sum());membership_deltas.append(newscore-score);assert newscore>=score-1e-6;M=newM
    trajectories.append(dict(site=site,seed=seed,initialization=init,trajectory=trajectory,best_train_fraction=best));variants['initial_'+init,None]=M0;variants['learned_'+init,None]=bestM
    for ns in [1,2,3]:
     null=rewire(bestM,757000+seed*10+ns);degree_errors.append(float((null.sum(0)-bestM.sum(0)).abs().max()));variants['learned_'+init+'_rewired',ns]=null
   scalar=model(val/st['input_scale'])*st['input_scale'];scalar_mse=float((scalar-val).square().sum()/den)
   for (kind,ns),M in variants.items():
    pairs=select_pairs(vz.square(),M);q=union_pairs(M,pairs);g=torch.zeros(len(vz),128,device='cuda').scatter_(1,pairs,1);exact=(g@M>0).float();union_errors.append(float((q-exact).abs().max()));assert bool((g.sum(-1)==2).all()) and bool((q.sum(-1)<=8).all());pred=((vz*q)@model.decoder+model.bias)*st['input_scale'];count=q.sum(-1).long();order=vz.abs().topk(8,dim=1).indices;smask=torch.zeros_like(vz).scatter_(1,order,(torch.arange(8,device='cuda')[None,:]<count[:,None]).float());spred=((vz*smask)@model.decoder+model.bias)*st['input_scale'];rec=dict(site=site,seed=seed,kind=kind,null_seed=ns,validation_write_mse=float((pred-val).square().sum()/den),scalar_top8_mse=scalar_mse,union_size_matched_scalar_mse=float((spred-val).square().sum()/den),validation_captured_energy_fraction=float((vz.square()*q).sum()/vz.square().sum()),mean_unique_atoms=float(q.sum(-1).mean()),min_unique_atoms=int(q.sum(-1).min()),fraction_selected_groups_overlap=float((q.sum(-1)<8).float().mean()),distinct_groups=len(torch.unique(M,dim=0)),unused_groups=int((torch.bincount(pairs.flatten(),minlength=128)==0).sum()));records.append(rec);saved[site,seed,kind,ns]=M.cpu()

   print('DONE',site,seed,[(r['kind'],round(r['validation_write_mse'],4)) for r in records[-10:]],flush=True)
 audits=dict(exact_union_max_abs_error=max(union_errors),rewire_atom_degree_max_error=max(degree_errors),hashes_unchanged={k:sha(p)==hashes[k] for k,p in paths.items()},finite=all(np.isfinite(r['validation_write_mse']) for r in records),minimum_fixed_assignment_membership_energy_change=min(membership_deltas),all_group_sizes_four=all(bool((M.sum(-1)==4).all()) for M in saved.values()),whole_model_forwards=0,test_evaluations=0);assert all(audits['hashes_unchanged'].values()) and audits['finite'] and audits['all_group_sizes_four'] and max(union_errors)==0 and max(degree_errors)==0;torch.save(dict(memberships=saved),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,trajectories=trajectories,audits=audits,elapsed_seconds=time.time()-start),indent=2));print(audits,flush=True)
if __name__=='__main__':main()

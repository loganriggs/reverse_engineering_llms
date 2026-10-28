"""H730: discrete anchored membership learning for captured coordinate energy."""
import json,time
from pathlib import Path
import torch,numpy as np
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_coactivation_group_swaps import sha
from run_bilin18_verified_head_sparse import SignedBSF
from run_bilin18_frozen_dictionary_groups import rewire
STEM='bilin18_learned_group_memberships'
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
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time();paths={'data':ART/'bilin18_verified_head_pca_data.pt','fits':ART/'bilin18_verified_head_write_pca_fits.pt','models':ART/'bilin18_verified_head_sparse_captures.pt','memberships':ART/'bilin18_frozen_dictionary_groups_captures.pt','sparse':ROOT/'scripts/run_bilin18_verified_head_sparse.py','rewiring':ROOT/'scripts/run_bilin18_frozen_dictionary_groups.py','runner':Path(__file__)};hashes={k:sha(p) for k,p in paths.items()};data=torch.load(paths['data'],weights_only=False);fits=torch.load(paths['fits'],weights_only=False)['fits'];models=torch.load(paths['models'],weights_only=False)['states'];old=torch.load(paths['memberships'],weights_only=False)['memberships']
 cfg=dict(experiment='H730',hashes=hashes,design='Frozen6scalar dictionaries, original2048train and256validationdocs, all16positions. Learn128anchored groups each4atoms by10rounds coordinate ascent on capturedraw coefficient-squared energy. Each groupretains its anchor atom. Fixedassignment membershipupdates accountforunion andchoose3highest uncoveredenergy neighbors. Assignment updates greedystart plus4alternating bestresponses; samegate for allvalidationconditions. Rawcorrelation andrandomanchored initializations, preservebesttraining-scored membershipset includinginitial. Evaluatebothinitials andbothlearned plus3degree-preservingrewires perlearnedfit. Full4096validationvectors, no tasklabels.',limitations='Energy proxy isnot reconstruction objective with nonorthogonaldecoder; fixeddictionary/anchor/groupbudgets mayrestrict representation. Localcoordinate ascent, notglobal optimum orjointdecodertraining. Training assignment restart canfinddifferentlocaloptima;besttrain snapshotretained. Validation usedonlyevaluation. Reusedvalidationnotfreshconfirmatory; reservedtestuntouched.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));records=[];trajectories=[];saved={};membership_deltas=[]
 for site,fit in fits.items():
  p=(data['activations'][site].reshape(-1,128).cuda()-fit['mean'].cuda().float())@fit['write_pca']['encode'].cuda().float()
  for seed in [7241,7242,7243]:
   st=models[site,'bsf1',seed];model=SignedBSF(1).cuda();model.load_state_dict(st['model']);z=model.encoder(p/st['input_scale']);w=z[:32768].square();vz=z[32768:];val=p[32768:];den=val.square().sum();variants={};gen=torch.Generator(device='cuda').manual_seed(seed+730000);rand=torch.rand(128,128,device='cuda',generator=gen);rand.fill_diagonal_(-1);randomM=torch.eye(128,device='cuda').scatter_(1,rand.topk(3,dim=1).indices,1)
   for init,M0 in [('raw',old[site,seed,'raw_overlap',None].cuda()),('random',randomM)]:
    M=M0.clone();best=-1.;bestM=M.clone();trajectory=[]
    for epoch in range(11):
     pairs=select_pairs(w,M);q=union_pairs(M,pairs);score=float((w*q).sum()/w.sum());trajectory.append(dict(epoch=epoch,train_captured_energy_fraction=score))
     if score>best:best=score;bestM=M.clone()
     if epoch==10:break
     newM=update_memberships(w,M,pairs);newscore=float((w*union_pairs(newM,pairs)).sum()/w.sum());membership_deltas.append(newscore-score);assert newscore>=score-1e-6;M=newM
    trajectories.append(dict(site=site,seed=seed,initialization=init,trajectory=trajectory,best_train_fraction=best));variants['initial_'+init,None]=M0;variants['learned_'+init,None]=bestM
    for ns in [1,2,3]:variants['learned_'+init+'_rewired',ns]=rewire(bestM,730000+seed*10+ns)
   scalar=model(val/st['input_scale'])*st['input_scale'];scalar_mse=float((scalar-val).square().sum()/den)
   for (kind,ns),M in variants.items():
    pairs=select_pairs(vz.square(),M);q=union_pairs(M,pairs);pred=((vz*q)@model.decoder+model.bias)*st['input_scale'];rec=dict(site=site,seed=seed,kind=kind,null_seed=ns,validation_write_mse=float((pred-val).square().sum()/den),scalar_top8_mse=scalar_mse,validation_captured_energy_fraction=float((vz.square()*q).sum()/vz.square().sum()),mean_unique_atoms=float(q.sum(-1).mean()),distinct_groups=len(torch.unique(M,dim=0)),unused_groups=int((torch.bincount(pairs.flatten(),minlength=128)==0).sum()));records.append(rec);saved[site,seed,kind,ns]=M.cpu()
   print('DONE',site,seed,[(r['kind'],round(r['validation_write_mse'],4)) for r in records[-10:]],flush=True)
 audits=dict(hashes_unchanged={k:sha(p)==hashes[k] for k,p in paths.items()},finite=all(np.isfinite(r['validation_write_mse']) for r in records),minimum_fixed_assignment_membership_energy_change=min(membership_deltas),all_group_sizes_four=all(bool((M.sum(-1)==4).all()) for M in saved.values()),whole_model_forwards=0,test_evaluations=0);assert all(audits['hashes_unchanged'].values()) and audits['finite'] and audits['all_group_sizes_four'];torch.save(dict(memberships=saved),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,trajectories=trajectories,audits=audits,elapsed_seconds=time.time()-start),indent=2));print(audits,flush=True)
if __name__=='__main__':main()

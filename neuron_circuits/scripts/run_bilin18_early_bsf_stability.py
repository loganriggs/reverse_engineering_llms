"""H684: seed stability of decoder subspaces and validation activation support."""
import json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_early_bsf_extended import BSF
STEM='bilin18_early_bsf_stability'
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time();paths={'extended':ART/'bilin18_early_bsf_extended.json','data':ART/'bilin18_early_bsf_data_captures.pt','trainer':ROOT/'scripts/run_bilin18_early_bsf_extended.py','runner':Path(__file__)};assert json.loads(paths['extended'].read_text())['status']=='complete';hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()};cfg=dict(experiment='H684',hashes=hashes,design='All4sites and3widths, seed681 matched to682 and683. Hungarian one-to-one matching by decoder subspace overlap mean squared principal cosine. Null: deterministic signed permutation of ambient coordinates of second decoder, then independently optimize assignment. Evaluate validation support Jaccard for both matchings. Test unused.',limitations='Seed stability is not semantic purity, manifold smoothness or causal fidelity. One ambient scramble per pair is a reference not significance test; allblocks and jointlyoccupied subsets reported. Bases allow rotations within blocks.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));data=torch.load(paths['data'],weights_only=False);results=[];checkpoint_hashes={}
 for site in ['value0','residual0','residual1','residual2']:
  raw=data['activations'][site][2048:2304].reshape(-1,1152).cuda()
  for rank in [1,4,16]:
   loaded={}
   for seed in [681,682,683]:
    p=ART/'early_bsf_extended'/f'{site}_r{rank}_seed{seed}.pt';checkpoint_hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();ck=torch.load(p,weights_only=False);m=BSF(1152,rank).cuda();m.load_state_dict(ck['state_dict']);x=(raw-ck['mean'].cuda())/ck['scale'].cuda();mask=torch.cat([m.encode(b)[1] for b in x.split(512)]).bool();D=m.decoder.reshape(-1,rank,1152);Q=torch.linalg.qr(D.transpose(1,2),mode='reduced').Q.transpose(1,2);loaded[seed]=(Q,mask);del m
   qa,ma=loaded[681];G=qa.shape[0]
   for seed in [682,683]:
    qb,mb=loaded[seed];rng=torch.Generator(device='cuda').manual_seed(684+seed+rank);perm=torch.randperm(1152,generator=rng,device='cuda');sign=2*torch.randint(2,(1152,),generator=rng,device='cuda')-1;entry=dict(site=site,block_dimension=rank,seed_a=681,seed_b=seed)
    for name,q in [('learned',qb),('scrambled',qb[:,:,perm]*sign)]:
     cross=qa.flatten(0,1)@q.flatten(0,1).T;score=cross.reshape(G,rank,G,rank).square().sum((1,3))/rank;sc=score.cpu().numpy();aa,bb=linear_sum_assignment(-sc);aidx=torch.tensor(aa,device='cuda');bidx=torch.tensor(bb,device='cuda');aact=ma[:,aidx];bact=mb[:,bidx];intersection=(aact&bact).sum(0).float();union=(aact|bact).sum(0).float();jac=intersection/union.clamp_min(1);eligible=(aact.sum(0)>=32)&(bact.sum(0)>=32);overlap=score[aidx,bidx];entry[name]=dict(mean_overlap=float(overlap.mean()),median_overlap=float(overlap.median()),mean_jaccard=float(jac.mean()),jointly_occupied_blocks=int(eligible.sum()),occupied_mean_overlap=float(overlap[eligible].mean()) if eligible.any() else None,occupied_mean_jaccard=float(jac[eligible].mean()) if eligible.any() else None,assignments=bb.tolist(),overlap=overlap.cpu().tolist(),jaccard=jac.cpu().tolist())
    results.append(entry);print('DONE',site,rank,seed,entry['learned']['mean_overlap'],entry['learned']['mean_jaccard'],flush=True)
 audits=dict(hashes_unchanged={k:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[k] for k,p in paths.items()},checkpoint_hashes_unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in checkpoint_hashes.items()),pairs=len(results),test_evaluations=0,finite=all(np.isfinite(x['learned']['mean_overlap']) for x in results));(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=results,checkpoint_hashes=checkpoint_hashes,audits=audits,elapsed_seconds=time.time()-start),indent=2));print(audits)
if __name__=='__main__':main()

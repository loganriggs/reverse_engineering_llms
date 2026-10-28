"""H839 frozen cross-seed geometry/support stability with lexical support null."""
import json,time
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from run_bilin18_balanced_shared_featurizers import Sparse,BalancedShared
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_early_local_geometry import sha
S='bilin18_exact_site_stability'
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time();paths={'fits':ART/'bilin18_balanced_shared_featurizers_captures.pt','data':ART/'bilin18_frozen_featurizer_confirmation_captures.pt','token_means':ART/'bilin18_operator_state_prediction_captures.pt','runner':Path(__file__)};hashes={k:sha(p) for k,p in paths.items()};cfg={'experiment':'H839','hashes':hashes,'design':'Match seed8351 to8352 via Hungarian maximum mean squared principal cosines. Scalar1D atoms, block8 subspaces, shared8-atom group subspaces. Evaluate unaltered512 H837 supports. Allgroups and jointly>=16 active groups retained.','nulls':'Five signed-permuted ambient decoder-coordinate references with independently optimized assignments. Also permute seedB support rows within exact token identity on tokens with>=4 occurrences, holding learned geometric match fixed. Five shuffles; preserve lexical occupancy.','scope':'Two seeds and reused confirmation activations; exploratory stability, not semantic recovery or new confirmation. Nulls descriptive, not formal significance. Fixed/shared graph definitions may differ across seeds.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');states=torch.load(paths['fits'],weights_only=False)['states'];data=torch.load(paths['data'],weights_only=False);means=torch.load(paths['token_means'],weights_only=False)['fits'];tids=data['token_ids'];unique,counts=torch.unique(tids,return_counts=True);repeated=torch.isin(tids,unique[counts>=4]);eligible_indices=repeated.nonzero().flatten();results=[];ortho=[];ranks=[]
 for l in [1,2]:
  f=means[l];b=data['captures'][l]['premlp'].double();u=b/(b.square().mean(1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();lookup={int(t):i for i,t in enumerate(f['token_ids'])};tm=torch.stack([f['token_shape_mean'][lookup[int(t)]] if int(t) in lookup else f['shape_mean'] for t in tids]);res=u-tm
  for kind in ['scalar48','scalar64','block8','shared_fixed','shared_learned']:
   loaded={}
   for seed in [8351,8352]:
    ck=states[l,kind,seed];m=BalancedShared(seed,kind=='shared_fixed').cuda() if kind.startswith('shared') else Sparse(8 if kind=='block8' else 1,48 if kind=='scalar48' else 64).cuda();m.load_state_dict(ck['state_dict']);rr=[m(x) for x in (res/ck['scale']).float().cuda().split(128)]
    if kind.startswith('shared'):
     M=rr[0]['membership'].detach();D=m.dictionary_matrix().T.double();groups=torch.stack([D[row.bool()] for row in M]);mask=torch.cat([r['gates'] for r in rr]).bool().cpu();rank=8
    else:
     rank=8 if kind=='block8' else 1;groups=m.decoder.double().reshape(-1,rank,1152);mask=torch.cat([r['support'] for r in rr])[:,::rank].bool().cpu()
    q=torch.linalg.qr(groups.transpose(1,2),mode='reduced').Q.transpose(1,2).cpu();ortho.append(float((q@q.transpose(1,2)-torch.eye(rank)).abs().max()));ranks.append(float(torch.linalg.svdvals(groups).min()));loaded[seed]=(q,mask);del m,rr
   qa,ma=loaded[8351];qb,mb=loaded[8352];G=len(qa)
   def match(q):
    cross=qa.flatten(0,1)@q.flatten(0,1).T;sim=cross.reshape(G,rank,G,rank).square().sum((1,3))/rank;ia,ib=linear_sum_assignment(-sim.numpy());return torch.tensor(ib),sim[ia,ib]
   def supports(a,b):return (a&b).sum(0).float()/(a|b).sum(0).clamp_min(1)
   matchidx,ov=match(qb);bm=mb[:,matchidx];jac=supports(ma,bm);occupied=(ma.sum(0)>=16)&(bm.sum(0)>=16);null=[]
   for rep in range(5):
    gen=torch.Generator().manual_seed(83900+100*l+rep);perm=torch.randperm(1152,generator=gen);sign=2*torch.randint(2,(1152,),generator=gen)-1;ni,no=match(qb[:,:,perm]*sign);nj=supports(ma,mb[:,ni]);null.append({'mean_overlap':float(no.mean()),'mean_support_Jaccard':float(nj.mean())})
   act=supports(ma[repeated],bm[repeated]);lex=[]
   for rep in range(5):
    gen=torch.Generator().manual_seed(839000+100*l+rep);perm=torch.arange(512)
    for token in unique[counts>=4]:
     ix=(tids==token).nonzero().flatten();perm[ix]=ix[torch.randperm(len(ix),generator=gen)]
    lex.append(float(supports(ma[repeated],bm[perm][repeated]).mean()))
   results.append({'layer':l,'kind':kind,'dimension':rank,'units':G,'mean_overlap':float(ov.mean()),'mean_support_Jaccard':float(jac.mean()),'occupied_units':int(occupied.sum()),'occupied_overlap':float(ov[occupied].mean()) if occupied.any() else None,'occupied_Jaccard':float(jac[occupied].mean()) if occupied.any() else None,'scrambled':null,'repeated_token_rows':int(repeated.sum()),'repeated_token_support_Jaccard':float(act.mean()),'within_token_shuffled_Jaccards':lex,'matching':matchidx.tolist(),'overlaps':ov.tolist(),'Jaccards':jac.tolist(),'seed_a_counts':ma.sum(0).tolist(),'seed_b_matched_counts':bm.sum(0).tolist()});print('DONE',l,kind,flush=True)
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'orthogonality_max_abs':max(ortho),'minimum_group_singular_value':min(ranks),'whole_model_forwards':0,'test_evaluations':0,'finite':all(np.isfinite(r['mean_overlap']) and np.isfinite(r['mean_support_Jaccard']) for r in results)};assert audits['hashes_unchanged'] and max(ortho)<1e-10 and min(ranks)>1e-8 and audits['finite'];(ART/f'{S}.json').write_text(json.dumps({'config':cfg,'records':results,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(audits,flush=True)
if __name__=='__main__':main()

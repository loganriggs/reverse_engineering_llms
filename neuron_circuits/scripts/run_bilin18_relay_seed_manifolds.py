"""H860: cross-seed subspace and activation stability; no model training or causal claims."""
import json,time
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_balanced_shared_featurizers import Sparse,BalancedShared
from run_bilin18_early_local_geometry import sha
S='bilin18_relay_seed_manifolds'
def similarity(q,z):
 return torch.cat([torch.einsum('gdr,hds->ghrs',b,z).square().sum((-1,-2))/q.shape[-1] for b in q.split(32)]).cpu().numpy()
def match(s):
 i,j=linear_sum_assignment(-s);assert np.array_equal(i,np.arange(len(i))) and len(set(j))==len(j);return j

def corrs(a,b):
 a=a-a.mean(0);b=b-b.mean(0);den=a.norm(dim=0)*b.norm(dim=0);return torch.where(den>1e-12,(a*b).sum(0)/den,torch.nan)
@torch.inference_mode()
def main():
 t=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest')
 paths={'fits':ART/'bilin18_relay_natural_featurizers_captures.pt','data':ART/'bilin18_relay_natural_data_captures.pt','confirmation':ART/'bilin18_norm_frozen_confirmation_captures.pt','panel':ART/'bilin18_frozen_featurizer_confirmation_captures.pt','directions':ART/'bilin18_relay_contrast_subspaces_fits.pt','runner':Path(__file__)};hashes={k:sha(v) for k,v in paths.items()}
 cfg={'experiment':'H860','hashes':hashes,'design':'All five architectures and both seeds. One-to-one group matching maximizes normalized projector overlap using dictionaries only. Evaluate decoded group-vector magnitude correlation across512 H859 natural occurrences. No activation-based matching. Postfit supervised task-axis overlap is diagnostic only.','groups':'Scalar atoms; native8-coordinate blocks;8-member shared groups. Shared activity is own gate times norm of union-coded coordinates decoded through group atoms. Overlapping group activities are not an additive attribution. No norm calibration: tests native geometry/routing, not calibrated output quality.','controls':'Eight random signed ambient-coordinate permutations of seed2 dictionary, preserving all within-dictionary geometry; recompute optimal matches each time. Activities are unchanged under these isometries. This is an orientation control, not a null for every training bias. Self/permutation/sign/within-block-rotation algebra checks.','scope':'Confirmation documents reused from H859; task axes discovery-supervised. Two seeds insufficient for population stability. No causal or semantic claim from agreement alone.'}
 (ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 states=torch.load(paths['fits'],weights_only=False)['states'];data=torch.load(paths['data'],weights_only=False);f=data['fits'][4];raw=torch.load(paths['confirmation'],weights_only=False)['raw_premlp'].double();panel=torch.load(paths['panel'],weights_only=False);tids=[int(panel['all_token_ids'][i,p]) for i,p in enumerate(panel['positions'])];lookup={int(x):i for i,x in enumerate(f['token_ids'])};means=torch.stack([f['token_shape_mean'][lookup[x]] if x in lookup else f['shape_mean'] for x in tids]);res=raw/(raw.square().mean(1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt()-means;axes=torch.load(paths['directions'],weights_only=False)['bases'];records=[];saved={};maxorth=0.;selferr=0.;invariance=0.
 gen=torch.Generator(device='cuda').manual_seed(860)
 for kind in ['scalar48','scalar64','block8','shared_fixed','shared_learned']:
  qs=[];activities=[]
  for seed in [8351,8352]:
   ck=states[4,kind,seed];m=BalancedShared(seed,kind=='shared_fixed').cuda() if kind.startswith('shared') else Sparse(8 if kind=='block8' else 1,48 if kind=='scalar48' else 64).cuda();m.load_state_dict(ck['state_dict']);outs=[m(z) for z in (res/ck['scale']).float().cuda().split(128)];code=torch.cat([o['code'] for o in outs]).double();D=(m.dictionary_matrix().T if kind.startswith('shared') else m.decoder).double()
   if kind.startswith('shared'):
    membership=outs[0]['membership'].bool();idx=torch.stack([r.nonzero().flatten() for r in membership]);gates=torch.cat([o['gates'] for o in outs]).double()
   else:
    rank=8 if kind=='block8' else 1;idx=torch.arange(512,device='cuda').reshape(-1,rank);gates=torch.ones(len(res),len(idx),device='cuda')
   groups=D[idx];U,sv,V=torch.linalg.svd(groups,full_matrices=False);assert bool((sv[:,-1]>sv[:,0]*1e-10).all());q=V.transpose(1,2);maxorth=max(maxorth,float((q.transpose(1,2)@q-torch.eye(q.shape[-1],device='cuda')).abs().max()));act=torch.einsum('ngi,gid->ngd',code[:,idx],groups).norm(dim=-1)*gates*ck['scale'];qs.append(q);activities.append(act.cpu());del m,outs
  q,z=qs;s=similarity(q,z);j=match(s);c=corrs(activities[0],activities[1][:,j]);selferr=max(selferr,float(abs(np.diag(similarity(q,q))-1).max()))
  rot=torch.linalg.qr(torch.randn(q.shape[-1],q.shape[-1],device='cuda',dtype=torch.double,generator=gen)).Q;perm=torch.randperm(len(q),device='cuda',generator=gen);check=similarity(q,q[perm]@rot);jj=match(check);invariance=max(invariance,float(abs(check[np.arange(len(jj)),jj]-1).max()))
  null=[]
  for n in range(8):
   p=torch.randperm(1152,device='cuda',generator=gen);sign=(torch.randint(2,(1152,),device='cuda',generator=gen)*2-1).double();sn=similarity(q,z[:,p]*sign[None,:,None]);jn=match(sn);cn=corrs(activities[0],activities[1][:,jn]);null.append({'mean_overlap':float(sn[np.arange(len(jn)),jn].mean()),'median_activity_correlation':float(cn.nanmedian())})
  axis=[]
  for family in ['agreement','sentiment']:
   v=axes[family][:,0].double().cuda();e1=(q.transpose(1,2)@v).square().sum(-1).cpu().numpy();e2=(z.transpose(1,2)@v).square().sum(-1).cpu().numpy();g=int(e1.argmax());axis.append({'family':family,'seed1_best_group':g,'matched_seed2_group':int(j[g]),'seed1_energy':float(e1[g]),'matched_seed2_energy':float(e2[j[g]]),'seed2_best_energy':float(e2.max()),'matched_activity_correlation':float(c[g])})
  rec={'kind':kind,'groups':len(q),'rank':q.shape[-1],'mean_matched_overlap':float(s[np.arange(len(j)),j].mean()),'median_matched_overlap':float(np.median(s[np.arange(len(j)),j])),'median_activity_correlation':float(c.nanmedian()),'active_in_both_groups':int(c.isfinite().sum()),'fraction_all_groups_overlap_above_half':float(np.mean(s[np.arange(len(j)),j]>.5)),'orientation_nulls':null,'task_axis_diagnostics':axis};records.append(rec);saved[kind]={'assignment':j,'similarities':s,'activity_correlations':c,'activities':activities};print(json.dumps(rec),flush=True)
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'orthogonality_max_abs':maxorth,'self_overlap_max_abs':selferr,'rotation_permutation_invariance_max_abs':invariance,'whole_model_forwards':0,'test_evaluations':0,'all_architectures':len(records)==5};assert audits['hashes_unchanged'] and max(maxorth,selferr,invariance)<1e-9 and audits['all_architectures'];torch.save(saved,ART/f'{S}_captures.pt');(ART/f'{S}.json').write_text(json.dumps({'config':cfg,'records':records,'audits':audits,'elapsed_seconds':time.time()-t},indent=2)+'\n');print(audits,flush=True)
if __name__=='__main__':main()

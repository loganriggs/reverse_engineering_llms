"""H714: training-matched sham neighborhoods and anchor/neighbor decomposition."""
import hashlib,json,time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_reader_operation_format import run
from run_bilin18_coactivation_group_swaps import sha,metrics
STEM='bilin18_group_neighbor_controls'
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time()
 paths={'train':ART/'bilin18_early_bsf_data_captures.pt','panel':ART/'bilin18_coactivation_replication_captures.pt','old':ART/'bilin18_coactivation_group_swaps_captures.pt','old_results':ART/'bilin18_coactivation_group_swaps.json','spectra':ART/'bilin18_early_rotation_spectra_captures.pt','rotations':ART/'bilin18_early_sparse_rotations_captures.pt','groups':ART/'bilin18_rotation_coactivation_groups_captures.pt','model':SNAPSHOT/'pytorch_model.bin','runner':Path(__file__),'metrics':ROOT/'scripts/run_bilin18_coactivation_group_swaps.py','loader':ROOT/'scripts/run_bilin18_qkov_causal_edges.py','forward':ROOT/'scripts/run_bilin18_reader_operation_format.py'}
 hashes={k:sha(p) for k,p in paths.items()};train=torch.load(paths['train'],weights_only=False);panel=torch.load(paths['panel'],weights_only=False);old=torch.load(paths['old'],weights_only=False);oldj=json.loads(paths['old_results'].read_text());sp=torch.load(paths['spectra'],weights_only=False)['states']['residual1'];Q=torch.load(paths['rotations'],weights_only=False)['rotations']['residual1','learned',7061].cuda();mu=sp['mean'].float().cuda();V=sp['eigenvectors'][:,:256].float().cuda();sd=sp['eigenvalues'][:256].float().cuda().sqrt();D=(V*sd)@Q;n=D.norm(dim=0);D/=n
 def encode(x):return (((x.cuda()-mu)@V)/sd)@Q*n
 c=encode(train['activations']['residual1'][:2048].reshape(-1,1152));mask=torch.zeros_like(c).scatter_(1,c.abs().topk(32,dim=1).indices,1);freq=mask.mean(0);csd=c.std(0);gram=(D.T@D).abs();groups=torch.load(paths['groups'],weights_only=False)['states']['residual1'];frozen=[]
 # Greedy successive disjoint minimum-cost matchings. Controls use training statistics only.
 for g in groups['groups']:
  a=g['anchor'];members=g['members'];neigh=members[1:];feat=torch.stack([freq.clamp_min(1e-6).log(),csd.clamp_min(1e-6).log(),gram[a]],dim=1);scale=feat.std(0).clamp_min(1e-6);Z=feat/scale;exclude=set(members);controls=[]
  for repeat in range(3):
   candidates=[i for i in range(256) if i not in exclude];cost=(Z[neigh,None,:]-Z[None,candidates,:]).square().sum(-1).cpu().numpy();ri,ci=linear_sum_assignment(cost);matched=[candidates[i] for i in ci];assert list(ri)==list(range(7));exclude.update(matched);controls.append(dict(members=matched,standardized_rms_error=float(np.sqrt(cost[ri,ci].mean()/3)),matched_features=feat[matched].cpu().tolist(),training_anchor_correlations=[float(cor[a,matched].mean()) for cor in groups['training_half_correlations']]))
  frozen.append(dict(anchor=a,members=members,target_features=feat[neigh].cpu().tolist(),feature_scales=scale.cpu().tolist(),controls=controls))
 cfg=dict(experiment='H714',hashes=hashes,design='Reuse all64H713directedpairs, all8frozengroups. Match each of7neighbors to a nonmember using training-only logTop32frequency, logsignedcoordinateSD, absdecoder cosinewithanchor; standardizebyacross-featureSD; Hungarianminimumcost. Three successivedisjoint controlsets/group, nooutcomeselection. Use H7135%fullgroupcap scale foranchoronlyandactualneighboronly, whosevectorssumexactlytofullgroupedit. Shamneighborcontributions rescaled toactualneighbor norm, then optionallyaddidenticalanchoredit. Primarycontrast actualneighbor donorKLgain minusmean3shamneighbors,32pairedclusters. Combinedshamgroups notfullnormmatched; reportnorms. Temperaturefit andobservedCEseparate.',limitations='Exploratory reuseH713pairs; controls approximately match3trainingmarginals, notcompletegeometry, correlations, pairdeltaangles ordownstreamJacobian. Distinctcontrolsetsnotindependent samples. No taskcorrectness ormanifoldproof. Componentnormsmayexceedfull5%cap duecancellation; measured explicitly. Originalreservedtestuntouched.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));(ART/f'{STEM}_frozen_controls.json').write_text(json.dumps(frozen,indent=2));del train,c,mask;raw=panel['activations']['residual1'].reshape(-1,1152).cuda();code=encode(raw);patches=[];sumerrors=[];normerrors=[]
 for p,oldpatch in zip(old['pairs'],old['patches']):
  a=p['anchor'];g=next(g for g in frozen if g['anchor']==a);delta=code[p['donor']]-code[p['receiver']];scale=min(1.,.05*p['receiver_residual_norm']/p['natural_coordinate_norm']);anchor=D[:,a]*delta[a]*scale;neigh=D[:,g['members'][1:]]@delta[g['members'][1:]]*scale;full=oldpatch['coordinate'].cuda()*scale;sumerrors.append(float((anchor+neigh-full).norm()/full.norm()));patch={'anchor':anchor,'neighbors':neigh}
  for i,ctrl in enumerate(g['controls']):
   v=D[:,ctrl['members']]@delta[ctrl['members']];assert v.norm()>1e-8;v*=neigh.norm()/v.norm();normerrors.append(float(abs(v.norm()-neigh.norm())/neigh.norm()));patch[f'sham_neighbors_{i}']=v;patch[f'sham_group_{i}']=anchor+v
  patches.append({k:v.cpu() for k,v in patch.items()})
 assert max(sumerrors)<1e-5 and max(normerrors)<1e-5;model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 buckets=defaultdict(list)
 for k,p in enumerate(old['pairs']):buckets[p['position']].append(k)
 records=[];logitrows=[];forwards=0
 for mode in patches[0]:
  for pos,inds in buckets.items():
   for st in range(0,len(inds),8):
    bb=inds[st:st+8];patch=torch.stack([patches[k][mode] for k in bb])
    def edit(m,args,out):
     x,bus=out;x=x.clone();x[:,-1]+=patch.to(x);return x,bus
    h=model.blocks[1].register_forward_hook(edit)
    try:logits,_=run(model,torch.stack([panel['all_token_ids'][old['pairs'][k]['receiver']//16,:pos+1] for k in bb]).cuda())
    finally:h.remove()
    forwards+=len(bb)
    for n,k in enumerate(bb):
     p=old['pairs'][k];r=dict(**p,mode=mode,edit_norm=float(patch[n].norm()),relative_edit_norm=float(patch[n].norm())/p['receiver_residual_norm'],**metrics(old['native_logits'][p['receiver']].cuda(),old['native_logits'][p['donor']].cuda(),logits[n],int(panel['all_token_ids'][p['receiver']//16,pos+1])));records.append(r);logitrows.append(logits[n].cpu())
  (ART/f'{STEM}.partial.json').write_text(json.dumps(dict(status='running',records=records),indent=2));print('DONE',mode,flush=True)
 audits=dict(hashes_unchanged={k:sha(p)==hashes[k] for k,p in paths.items()},finite=all(torch.isfinite(torch.tensor([r['donor_kl_gain'],r['observed_next_token_ce_change']])).all().item() for r in records),anchor_plus_neighbors_replay_max_relative=max(sumerrors),sham_neighbor_norm_match_max_relative=max(normerrors),whole_model_forwards=forwards,test_evaluations=0)
 assert all(audits['hashes_unchanged'].values()) and audits['finite'];torch.save(dict(patches=patches,edited_logits=torch.stack(logitrows)),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,frozen_controls=frozen,records=records,audits=audits,elapsed_seconds=time.time()-start),indent=2));print(audits,flush=True)
if __name__=='__main__':main()

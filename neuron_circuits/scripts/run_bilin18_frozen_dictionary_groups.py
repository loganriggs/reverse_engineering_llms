"""H726: training-derived overlapping supports on frozen scalar dictionaries."""
import json,time
from pathlib import Path
import torch,numpy as np
from scipy.optimize import linear_sum_assignment
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_coactivation_group_swaps import sha
from run_bilin18_verified_head_sparse import SignedBSF
from feature_structures.models import union_support
STEM='bilin18_frozen_dictionary_groups'
def greedy_union(z,M,budget=2):
 q=torch.zeros_like(z);g=torch.zeros(len(z),len(M),device=z.device,dtype=z.dtype)
 for _ in range(budget):
  scores=(z.square()*(1-q))@M.T;scores=scores.masked_fill(g.bool(),-torch.inf);i=scores.argmax(-1);g.scatter_(1,i[:,None],1);q=torch.maximum(q,M[i])
 return q,g

def rewire(M,seed):
 rng=np.random.default_rng(seed);arr=M.cpu().numpy().astype(np.int64);edges=np.argwhere(arr);success=0
 for attempt in range(200000):
  a,b=rng.integers(0,len(edges),2);g,i=edges[a];h,k=edges[b]
  if g==h or i==k or arr[g,k] or arr[h,i]:continue
  arr[g,i]=arr[h,k]=0;arr[g,k]=arr[h,i]=1;edges[a,1]=k;edges[b,1]=i;success+=1
  if success==10000:break
 result=torch.tensor(arr,dtype=M.dtype,device=M.device);assert success==10000 and torch.equal(result.sum(0),M.sum(0)) and torch.equal(result.sum(1),M.sum(1));return result
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time();paths={'data':ART/'bilin18_verified_head_pca_data.pt','tokens':ART/'bilin18_early_bsf_data_captures.pt','fits':ART/'bilin18_verified_head_write_pca_fits.pt','models':ART/'bilin18_verified_head_sparse_captures.pt','sparse_implementation':ROOT/'scripts/run_bilin18_verified_head_sparse.py','union_implementation':ROOT/'scripts/feature_structures/models.py','runner':Path(__file__)};hashes={k:sha(p) for k,p in paths.items()};data=torch.load(paths['data'],weights_only=False);tokens=data['all_token_ids'][:,data['positions']];fits=torch.load(paths['fits'],weights_only=False)['fits'];models=torch.load(paths['models'],weights_only=False)['states'];cfg=dict(experiment='H726',hashes=hashes,design='Freezeall6H724scalarencoder/decoders. FromtrainingTop8supports construct128anchorgroups of4atoms:anchor+3neighbors maximizingminimumcorrelationacross1024documenthalves. Two versions:rawbinarysupport correlation,orwithin(token,position)residualizedsupport correlation with>=8examples/stratum. No validation/task membershipfit. For eachgrouping3degree-preservingrewiringcontrols preserve128groups,4members/group,allatomdegrees,512memberships. Additionalrandomdisjoint32groupsof4,128memberships;differentgatebudgetcapacityexplicit. Select2groupsgreedily byuncoveredrawcoordinateenergy; exactuniondecodes sharedatomsonce, <=8uniqueactives. Samefrozenencoderrawcoordinatesanddecoder for allconditions; no group-gatelearning. EvaluatevalidationwriteMSE, occupancy, originalTop8coverage andtrainhalfmembershipstability.',limitations='Heuristicmembershipinferenceandgreedygates,notjointoptimizationorexactlasso. Decoderatomsnonorthogonal soenergyselectionnotminimumMSE. Fixed2groupsnominal8slots;overlapreducesuniqueactivecount. Disjointcontrolhasfewergroups/memberships. Rawcoactivationmaybelexical;conditionalcontrolonlyremovestoken/positionmeans. No manifold/semantic claim; originaltestuntouched.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));records=[];states={};group_info=[];dictionaries={};exacterrors=[]
 def corr(y,keys=None):
  if keys is not None:
   u,inv,counts=torch.unique(keys,return_inverse=True,return_counts=True);sums=torch.zeros(len(u),128,device='cuda').index_add_(0,inv,y);y=(y-sums[inv]/counts[inv,None])[counts[inv]>=8]
  else:y=y-y.mean(0)
  energy=y.square().sum(0);c=y.T@y/(energy[:,None]*energy[None,:]).sqrt().clamp_min(1e-12);return c,dict(rows=len(y),variable_atoms=int((energy>1e-8).sum()))
 def membership(c):
  c=c.clone();c.fill_diagonal_(-2);ids=c.topk(3,dim=1).indices;M=torch.eye(128,device='cuda').scatter_(1,ids,1);return M
 for site,fit in fits.items():
  p=(data['activations'][site].reshape(-1,128).cuda()-fit['mean'].cuda().float())@fit['write_pca']['encode'].cuda().float();train=p[:32768];val=p[32768:36864]
  for seed in [7241,7242,7243]:
   state=models[site,'bsf1',seed];model=SignedBSF(1).cuda();model.load_state_dict(state['model']);dictionaries[site,seed]=model.decoder.cpu();z=model.encoder(train/state['input_scale']);_,support=model.encode(train/state['input_scale']);vz=model.encoder(val/state['input_scale']);_,vmask=model.encode(val/state['input_scale']);den=val.square().sum();basepred=model(val/state['input_scale'])*state['input_scale'];records.append(dict(site=site,seed=seed,kind='scalar_top8',null_seed=None,validation_write_relative_mse=float((basepred-val).square().sum()/den),mean_unique_active_atoms=8.,top8_support_recall=1.));members={}
   for kind in ['raw_overlap','conditional_overlap']:
    cs=[];cas=[]
    for lo,hi in [(0,16384),(16384,32768)]:
     keys=tokens[:2048].flatten()[lo:hi].cuda()*16+torch.arange(16,device='cuda').repeat(1024) if kind=='conditional_overlap' else None;c,ca=corr(support[lo:hi],keys);cs.append(c);cas.append(ca)
    score=torch.minimum(*cs);M=membership(score);ma=membership(cs[0]);mb=membership(cs[1]);inter=(ma*mb).sum(-1);halfj=float((inter/(8-inter)).mean());degrees=M.sum(0);positive=int(((M*score).sum(-1)-score.diag()>0).sum());group_info.append(dict(site=site,seed=seed,kind=kind,half_membership_mean_jaccard=halfj,half_audits=cas,atom_degree_min=int(degrees.min()),atom_degree_max=int(degrees.max()),distinct_groups=len(torch.unique(M,dim=0))));members[kind,None]=M
    for ns in [1,2,3]:members[kind+'_rewired',ns]=rewire(M,726000+seed*10+ns)
   gen=torch.Generator(device='cuda').manual_seed(seed+726);perm=torch.randperm(128,generator=gen,device='cuda').reshape(32,4);members['random_disjoint',None]=torch.zeros(32,128,device='cuda').scatter_(1,perm,1)
   for (kind,ns),M in members.items():
    q,g=greedy_union(vz,M);assert torch.all(g.sum(-1)==2) and torch.all(q.sum(-1)<=8) and torch.all(M.sum(-1)==4);exact=union_support(g[:32],M);exacterrors.append(float((exact-q[:32]).abs().max()));pred=((vz*q)@model.decoder+model.bias)*state['input_scale'];mse=float((pred-val).square().sum()/den);rec=dict(site=site,seed=seed,kind=kind,null_seed=ns,validation_write_relative_mse=mse,mean_unique_active_atoms=float(q.sum(-1).mean()),min_unique_active_atoms=int(q.sum(-1).min()),top8_support_recall=float((q*vmask).sum()/vmask.sum()),dead_groups=int((g.sum(0)==0).sum()),groups=len(M),memberships=int(M.sum()),active_groups=2);records.append(rec);states[site,seed,kind,ns]=M.cpu()
   print('GROUPS',site,seed,[(r['kind'],round(r['validation_write_relative_mse'],4)) for r in records[-9:]],flush=True)
 stability=[]
 for site in fits:
  for sa,sb in [(7241,7242),(7241,7243),(7242,7243)]:
   Da=dictionaries[site,sa];Db=dictionaries[site,sb];sim=(Da@Db.T).abs();ia,ib=linear_sum_assignment(-sim.numpy());assert np.array_equal(ia,np.arange(128))
   for kind in ['raw_overlap','conditional_overlap']:
    A=states[site,sa,kind,None];B=states[site,sb,kind,None][:,ib];inter=A@B.T;jac=inter/(8-inter);ga,gb=linear_sum_assignment(-jac.numpy());stability.append(dict(site=site,seeds=[sa,sb],kind=kind,matched_atom_cosine=float(sim[ia,ib].mean()),matched_group_membership_jaccard=float(jac[ga,gb].mean())))
 audits=dict(hashes_unchanged={k:sha(p)==hashes[k] for k,p in paths.items()},finite=all(np.isfinite(r['validation_write_relative_mse']) for r in records),exact_union_max_abs_error=max(exacterrors),degree_preserving_rewire_assertions=True,whole_model_forwards=0,test_evaluations=0);assert all(audits['hashes_unchanged'].values()) and audits['finite'] and max(exacterrors)==0;torch.save(dict(memberships=states),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,group_info=group_info,stability=stability,audits=audits,elapsed_seconds=time.time()-start),indent=2));print(audits,flush=True)
if __name__=='__main__':main()

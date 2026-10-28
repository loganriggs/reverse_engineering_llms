"""H711: train-derived overlapping groups from token/position-conditioned activity."""
import torch,json,time,hashlib
from pathlib import Path
from run_bilin18_diverse_causal_tasks import ROOT,ART
STEM='bilin18_rotation_coactivation_groups'
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time();paths={'data':ART/'bilin18_early_bsf_data_captures.pt','spectra':ART/'bilin18_early_rotation_spectra_captures.pt','rotations':ART/'bilin18_early_sparse_rotations_captures.pt','atlas':ART/'bilin18_rotation_stability_context.json','runner':Path(__file__)};hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()};data=torch.load(paths['data'],weights_only=False);specs=torch.load(paths['spectra'],weights_only=False)['states'];rotations=torch.load(paths['rotations'],weights_only=False)['rotations'];atlas=json.loads(paths['atlas'].read_text());cfg=dict(experiment='H711',hashes=hashes,design='H706seed7061 Top32binary activity. Residualize within joint(token,sampledposition)strata separately in trainingdocument halves andvalidation; require>=8examples/stratum. For eight H707stable anchors, select7neighbors maximizing min(traininghalf conditionalcorrelations), requirepositive score. Validate anchor/group conditionalcorrelation against100independent within-stratum per-feature shuffles. Overlapping shared-direction groups; no semantic/tasklabels.',limitations='Anchorselection previously usedvalidationstability, neighbors use trainingonly. Validation reused exploratory; null conditionalexchangeability ignores otherdocumentdependencies. Fixed8direction neighborhoods are candidate groups, not learned global group gates or demonstrated manifolds.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));records=[];siteaudits=[];states={}
 def residualize(y,keys):
  unique,inv,counts=torch.unique(keys,return_inverse=True,return_counts=True);sums=torch.zeros(len(unique),256,device='cuda').index_add_(0,inv,y);means=sums/counts[:,None];keep=counts[inv]>=8;return (y-means[inv])[keep],inv[keep],dict(eligible_examples=int(keep.sum()),eligible_strata=int((counts>=8).sum()))
 def correlation(r):
  energy=r.square().sum(0);valid=energy>1e-8;cov=r.T@r;cor=cov/(energy[:,None]*energy[None,:]).sqrt().clamp_min(1e-12);cor[~valid,:]=0;cor[:,~valid]=0;return cor,valid
 for site in ['value0','residual1']:
  sp=specs[site];mu=sp['mean'].cuda().float();V=sp['eigenvectors'][:,:256].cuda().float();sd=sp['eigenvalues'][:256].cuda().float().sqrt();Q=rotations[site,'learned',7061].cuda();norm=((V*sd)@Q).norm(dim=0);pieces=[];strata=[];aud=[]
  for lo,hi in [(0,1024),(1024,2048),(2048,2304)]:
   x=data['activations'][site][lo:hi].reshape(-1,1152).cuda()-mu;code=(((x@V)/sd)@Q)*norm;idx=code.abs().topk(32,dim=-1).indices;y=torch.zeros_like(code).scatter_(1,idx,1);tokens=data['token_ids'][lo:hi].flatten().cuda();positions=torch.arange(16,device='cuda').repeat(hi-lo);r,ids,a=residualize(y,tokens*16+positions);pieces.append(r);strata.append(ids);aud.append(a)
  ca,va=correlation(pieces[0]);cb,vb=correlation(pieces[1]);cv,vv=correlation(pieces[2]);anchors=[r['direction'] for r in atlas['lexical'] if r['site']==site];siteaudits.append(dict(site=site,splits=aud,valid_directions=[int(v.sum()) for v in [va,vb,vv]],max_residual_abs=float(max(r.abs().max() for r in pieces))))
  if not (va&vb).any():print(site,'NO CONDITIONAL VARIANCE',flush=True);continue
  groups=[]
  for anchor in anchors:
   score=torch.minimum(ca[anchor],cb[anchor]);score[anchor]=-2;score[~(va&vb)]=-2;neighbors=score.topk(7).indices;positive=bool((score[neighbors]>0).all())
   if not positive:continue
   members=[anchor]+neighbors.tolist();within=[(members[i],members[k]) for i in range(8) for k in range(i+1,8)];groups.append(dict(anchor=anchor,members=members,training_half_anchor_correlations=[ca[anchor,neighbors].cpu().tolist(),cb[anchor,neighbors].cpu().tolist()],validation_anchor_correlation=float(cv[anchor,neighbors].mean()),validation_all_pair_correlation=float(torch.stack([cv[a,b] for a,b in within]).mean()),all_pairs=within,null_anchor=[],null_all=[]))
  # Each direction is independently permuted inside each eligible validation stratum.
  r=pieces[2];keys=strata[2];rowgroups=[(keys==k).nonzero().flatten() for k in keys.unique()];gen=torch.Generator(device='cuda').manual_seed(711);columns=torch.arange(256,device='cuda')[None,:]
  for repeat in range(100):
   shuffled=r.clone()
   for rows in rowgroups:
    perm=torch.rand(len(rows),256,generator=gen,device='cuda').argsort(0);shuffled[rows]=r[rows[perm],columns]
   null,_=correlation(shuffled)
   for g in groups:
    g['null_anchor'].append(float(null[g['anchor'],g['members'][1:]].mean()));g['null_all'].append(float(torch.stack([null[a,b] for a,b in g['all_pairs']]).mean()))
  for g in groups:
   null=torch.tensor(g.pop('null_anchor'));allnull=torch.tensor(g.pop('null_all'));g.pop('all_pairs');rec=dict(site=site,**g,null_anchor_mean=float(null.mean()),null_anchor_95_interval=torch.quantile(null,torch.tensor([.025,.975])).tolist(),null_all_mean=float(allnull.mean()),null_all_95_interval=torch.quantile(allnull,torch.tensor([.025,.975])).tolist(),null_anchor_at_least_observed=int((null>=g['validation_anchor_correlation']).sum()));records.append(rec);print(rec,flush=True)
  states[site]=dict(groups=[dict(anchor=g['anchor'],members=g['members']) for g in groups],training_half_correlations=[ca.cpu(),cb.cpu()],validation_correlation=cv.cpu())
 audits=dict(hashes_unchanged={k:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[k] for k,p in paths.items()},finite=all(torch.isfinite(torch.tensor(r['validation_anchor_correlation'])).item() for r in records),test_evaluations=0,whole_model_forwards=0);assert all(audits['hashes_unchanged'].values()) and audits['finite'];torch.save(dict(states=states),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,site_audits=siteaudits,audits=audits,elapsed_seconds=time.time()-start),indent=2));print(audits,flush=True)
if __name__=='__main__':main()

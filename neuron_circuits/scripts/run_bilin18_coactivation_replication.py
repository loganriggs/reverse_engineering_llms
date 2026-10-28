"""H712: frozen-group replication on document/window-disjoint corpus contexts."""
import torch,json,time,hashlib
import numpy as np
from pathlib import Path
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
STEM='bilin18_coactivation_replication'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time();paths={'old_manifest':ART/'bilin18_early_bsf_data_manifest.json','groups':ART/'bilin18_rotation_coactivation_groups.json','spectra':ART/'bilin18_early_rotation_spectra_captures.pt','rotations':ART/'bilin18_early_sparse_rotations_captures.pt','model':SNAPSHOT/'pytorch_model.bin','runner':Path(__file__)};hashes={k:sha(p) for k,p in paths.items()};old=json.loads(paths['old_manifest'].read_text());exclude={r['document_hash'] for r in old['records']};windows={r['window_sha256_int64'] for r in old['records']};pool={};sources={}
 for p in sorted(ART.glob('*documents.manifest.json')):
  j=json.loads(p.read_text());rs=j.get('records',[]);arr=p.with_name(p.name.replace('.manifest.json','.npy'))
  if not rs or not arr.exists() or not all('document_token_sha256' in r for r in rs):continue
  x=np.load(arr,mmap_mode='r')
  if x.ndim!=2 or len(x)!=len(rs) or x.shape[1]<128:continue
  sources[str(p)]=sha(p);sources[str(arr)]=sha(arr)
  for i,r in enumerate(rs):pool.setdefault(r['document_token_sha256'],dict(document_hash=r['document_token_sha256'],source=str(arr),source_row=i))
 arrays={};selected=[];tokens=[];seen=set(windows)
 for h in sorted(set(pool)-exclude,key=lambda h:hashlib.sha256(('H712:'+h).encode()).hexdigest()):
  r=pool[h].copy()
  if r['source'] not in arrays:arrays[r['source']]=np.load(r['source'],mmap_mode='r')
  ids=np.asarray(arrays[r['source']][r['source_row'],:128],dtype=np.int64);wh=hashlib.sha256(ids.tobytes()).hexdigest()
  if wh in seen:continue
  seen.add(wh);r['window_sha256_int64']=wh;selected.append(r);tokens.append(ids)
  if len(selected)==2048:break
 assert len(selected)==2048 and not({r['document_hash'] for r in selected}&exclude);tokens=torch.tensor(np.stack(tokens));positions=list(range(7,128,8));cfg=dict(experiment='H712',hashes=hashes,design='Freeze all8 H711groups. Select2048existing corpus documents by H712hashsort excluding allH681train/val/test document hashes and128token windowhashes. First2blocks capture value0/residual1at16positions. Same token+position conditioning,>=8examples/stratum.199within-stratum per-feature shuffle controls, report raw and8group max-statistic tail counts. No refit or neighbor selection.',limitations='Representation-fitting-disjoint but reused researchcorpus, not model-training holdout. Hashdisjointness doesnotexclude near duplicates. Groups/metric fixed before newactivationevaluation. Shuffle ignores otherdocumentdependencies/globalTop32constraint; descriptive inference.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));(ART/f'{STEM}_manifest.json').write_text(json.dumps(dict(records=selected,source_hashes=sources,unique_pool_documents=len(pool)),indent=2));model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 out={s:torch.empty(2048,16,1152) for s in ['value0','residual1']};pidx=torch.tensor(positions,device='cuda')
 for st in range(0,2048,16):
  ids=tokens[st:st+16].cuda();x0=torch.nn.functional.rms_norm(model.embedding(ids),(1152,));x=x0;bus=None
  for layer in range(2):
   x,bus=model.blocks[layer](x,bus,x0)
   if layer==0:out['value0'][st:st+len(ids)]=bus.reshape(len(ids),128,1152)[:,pidx].cpu()
   else:out['residual1'][st:st+len(ids)]=x[:,pidx].cpu()
  if st%512==0:print('CAPTURE',st,flush=True)
 del model;torch.cuda.empty_cache();torch.save(dict(activations=out,all_token_ids=tokens,token_ids=tokens[:,positions],positions=positions),ART/f'{STEM}_captures.pt');specs=torch.load(paths['spectra'],weights_only=False)['states'];rots=torch.load(paths['rotations'],weights_only=False)['rotations'];groups=json.loads(paths['groups'].read_text())['records'];records=[];siteaudits=[]
 def corr(r):
  e=r.square().sum(0);v=e>1e-8;c=(r.T@r)/(e[:,None]*e[None,:]).sqrt().clamp_min(1e-12);c[~v,:]=0;c[:,~v]=0;return c,v
 for site in ['value0','residual1']:
  sp=specs[site];mu=sp['mean'].cuda().float();V=sp['eigenvectors'][:,:256].cuda().float();sd=sp['eigenvalues'][:256].cuda().float().sqrt();Q=rots[site,'learned',7061].cuda();norm=((V*sd)@Q).norm(dim=0);x=out[site].reshape(-1,1152).cuda()-mu;code=(((x@V)/sd)@Q)*norm;idx=code.abs().topk(32,dim=-1).indices;y=torch.zeros_like(code).scatter_(1,idx,1);keys=tokens[:,positions].flatten().cuda()*16+torch.arange(16,device='cuda').repeat(2048);unique,inv,counts=torch.unique(keys,return_inverse=True,return_counts=True);sums=torch.zeros(len(unique),256,device='cuda').index_add_(0,inv,y);keep=counts[inv]>=8;r=(y-sums[inv]/counts[inv,None])[keep];ids=inv[keep];c,v=corr(r);siteaudits.append(dict(site=site,eligible_examples=int(keep.sum()),eligible_strata=int((counts>=8).sum()),valid_directions=int(v.sum()),max_residual_abs=float(r.abs().max())))
  if site=='value0':assert float(r.abs().max())==0;continue
  rowgroups=[(ids==k).nonzero().flatten() for k in ids.unique()];cols=torch.arange(256,device='cuda')[None,:];gen=torch.Generator(device='cuda').manual_seed(712);nulls=[]
  for repeat in range(199):
   shuffled=r.clone()
   for rows in rowgroups:
    perm=torch.rand(len(rows),256,generator=gen,device='cuda').argsort(0);shuffled[rows]=r[rows[perm],cols]
   nc,_=corr(shuffled);nulls.append([float(nc[g['anchor'],g['members'][1:]].mean()) for g in groups])
  null=torch.tensor(nulls);maxnull=null.max(-1).values
  for k,g in enumerate(groups):
   a=g['anchor'];members=g['members'];obs=float(c[a,members[1:]].mean());pairs=[c[members[i],members[t]] for i in range(8) for t in range(i+1,8)];rec=dict(anchor=a,members=members,old_validation_anchor_correlation=g['validation_anchor_correlation'],replication_anchor_correlation=obs,replication_all_pair_correlation=float(torch.stack(pairs).mean()),null_95_interval=torch.quantile(null[:,k],torch.tensor([.025,.975])).tolist(),null_at_least_observed=int((null[:,k]>=obs).sum()),max_statistic_at_least_observed=int((maxnull>=obs).sum()));records.append(rec);print(rec,flush=True)
 audits=dict(hashes_unchanged={k:sha(p)==hashes[k] for k,p in paths.items()},sources_unchanged=all(sha(Path(p))==h for p,h in sources.items()),document_disjoint=True,window_disjoint=True,documents=2048,new_documents=0,two_block_prefix_forwards=2048,whole_model_forwards=0,test_evaluations=0,finite=all(torch.isfinite(torch.tensor(r['replication_anchor_correlation'])).item() for r in records));assert all(audits['hashes_unchanged'].values()) and audits['sources_unchanged'] and audits['finite'];(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,site_audits=siteaudits,audits=audits,elapsed_seconds=time.time()-start),indent=2));print(audits,flush=True)
if __name__=='__main__':main()

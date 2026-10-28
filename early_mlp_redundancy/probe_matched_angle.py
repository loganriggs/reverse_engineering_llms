"""M6: fixed MLP1 matched norm/angle random-direction reference."""
import json,hashlib
from pathlib import Path
import torch
from inspect_model import load_model,trace
from probe_token_table_causal import score
P=Path(__file__).resolve().parent;A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition');SEEDS=[17,29,43]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');torch.set_grad_enabled(False)
 cp=P/'validation/natural_token_context_cohort.json';m4p=P/'validation/natural_token_context.json';m5p=P/'validation/token_table_causal.json';c=json.loads(cp.read_text());m4=json.loads(m4p.read_text());m5=json.loads(m5p.read_text());cache=Path(m4['cache']);assert sha(cache)==m4['cache_sha256'];assert sha(cp)==m4['cohort_sha256']
 protocol=dict(source_sha256=sha(Path(__file__)),cohort_sha256=sha(cp),M4_sha256=sha(m4p),M5_sha256=sha(m5p),seeds=SEEDS,design='MLP1 only, same64 reused documents, opposite-half token tables, covered positions and bias as M5. Each random product has same projection onto native product unit direction and same perpendicular norm as table product, hence same norm and local squared error. Gaussian vectors drawn independently per position, projected orthogonal to native and normalized. CPU float64 geometry, float32 writes. Seeds17/29/43; perdocument seed=seed+10000*document. All seeds reported. Native/table replay plus3random arms perdocument,320forwards. No tuning, new data or holdout.',limitations='Isotropic ambient directions may leave data manifold and remove cross-position correlations. Reference is not a semantic null or causal feature. Compare distributions of paired document effects; three draws do not establish a population frontier. Stop fixed perturbation grid after this comparison.')
 pp=P/'validation/matched_angle_protocol.json';pp.write_text(json.dumps(protocol,indent=2)+'\n');s=torch.load(cache,weights_only=True);tok=s['tokens'];docs=s['documents'];uni=s['unique_indices'];y=s['Y'][1];tables={};covered={}
 for half in [0,1]:
  tr=uni[docs[uni]//32!=half];types,inv,ns=torch.unique(tok[tr],return_inverse=True,return_counts=True);table=torch.zeros(50257,1152);mu=torch.zeros(len(types),1152,dtype=torch.float64).index_add_(0,inv,y[tr].double())/ns[:,None];table[types]=mu.float();tables[half]=table;covered[half]=types
 model,receipt=load_model();assert receipt.weights_sha256==m4['checkpoint']['weights_sha256'];device=next(model.parameters()).device;mlp=model.transformer.h[1].mlp;records=[];samples=[];checks=[];calls=0;replay=[]
 for d,ids0 in enumerate(c['token_ids']):
  ids=torch.tensor([ids0],device=device);native,cap=trace(model,ids);calls+=1;base=score(native,native,ids);v=mlp.Down(cap[1]['products']).cpu();assert torch.equal(v[0],y[d*128:(d+1)*128]);mask=torch.isin(torch.tensor([ids0]),covered[d//32]);t=tables[d//32][torch.tensor(ids0)][None];vd=v.double();td=t.double();vn=vd.norm(dim=-1,keepdim=True);assert vn.min()>1e-12;u=vd/vn;parallel=(td*u).sum(-1,keepdim=True);perp=(td-parallel*u).norm(dim=-1,keepdim=True)
  sample=dict(document=d,ids=ids.cpu(),native=native.cpu(),v=v,table=t,mask=mask,bias=mlp.Down_bias.cpu(),conditions=[]) if d in [0,32] else None
  for seed in [None]+SEEDS:
   if seed is None:q=t;gaussian=None;kind='table'
   else:
    gen=torch.Generator().manual_seed(seed+10000*d);gaussian=torch.randn(v.shape,generator=gen,dtype=torch.float64);g=gaussian-(gaussian*u).sum(-1,keepdim=True)*u;assert g.norm(dim=-1).min()>1e-12;e=g/g.norm(dim=-1,keepdim=True);q=(parallel*u+perp*e).float();kind='random'
    qd=q.double();scale=td.norm(dim=-1).clamp_min(1e-12);normerr=((qd.norm(dim=-1)-td.norm(dim=-1)).abs()/scale)[mask];doterr=(((qd*u).sum(-1)-(td*u).sum(-1)).abs()/scale)[mask];target=(td-vd).square().sum(-1);distanceerr=(((qd-vd).square().sum(-1)-target).abs()/target.clamp_min(1e-12))[mask];check=dict(document=d,seed=seed,max_norm_relative=float(normerr.max()),max_projection_relative=float(doterr.max()),max_squared_distance_relative=float(distanceerr.max()));assert max(check[k] for k in ['max_norm_relative','max_projection_relative','max_squared_distance_relative'])<1e-5;checks.append(check)
   qdev=q.to(device);mdev=mask.to(device)
   def replace(i,z,write):return torch.where(mdev[:,:,None],qdev+mlp.Down_bias,write) if i==1 else write
   out,cc=trace(model,ids,replace=replace);calls+=1;assert torch.equal(cc[1]['effective_write'][~mdev],cap[1]['native_write'][~mdev]);met=score(native,out,ids);rec=dict(document=d,kind=kind,seed=seed,native_CE=base['CE'],delta_CE=met['CE']-base['CE'],**met);records.append(rec)
   if seed is None:
    old=next(x for x in m5['records'] if x['document']==d and x['layer']==1 and x['kind']=='table');err=max(abs(rec[k]-old[k]) for k in ['KL','CE','top1_agreement']);assert err<1e-10;replay.append(err)
   if sample is not None:sample['conditions'].append(dict(seed=seed,q=q,gaussian=gaussian,output=out.cpu(),effective_write=cc[1]['effective_write'].cpu()))
  if sample is not None:samples.append(sample)
  if d%8==7:print('DOCUMENTS',d+1,flush=True)
 summaries=[]
 for seed in [None]+SEEDS:
  rows=[r for r in records if r['seed']==seed];summary=dict(seed=seed,**{k:sum(r[k] for r in rows)/64 for k in ['KL','CE','delta_CE','top1_agreement']})
  if seed is not None:
   diffs=[x['KL']-next(r['KL'] for r in records if r['document']==x['document'] and r['seed'] is None) for x in rows];summary.update(documents_random_KL_above_table=sum(v>0 for v in diffs),mean_paired_KL_difference=sum(diffs)/64)
  summaries.append(summary)
 sp=A/'emlp_m6_matched_angle_samples.pt';torch.save(samples,sp);out=dict(status='completed',protocol_sha256=sha(pp),forward_count=calls,M5_replay_max_error=max(replay),records=records,geometry_checks=checks,summaries=summaries,samples_path=str(sp),samples_sha256=sha(sp));(P/'validation/matched_angle.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(summaries,indent=2),flush=True)
if __name__=='__main__':main()

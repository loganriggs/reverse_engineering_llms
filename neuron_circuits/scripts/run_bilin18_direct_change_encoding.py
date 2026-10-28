"""H1044 frozen endpoint-difference versus direct-change sparse representations."""
from pathlib import Path
import sys,json,hashlib,ast,numpy as np,torch
from torch import nn
R=Path('/home/loganriggs/Coding/neuron_circuits');A=R/'artifacts/bilin18_tensor_decomposition';S='bilin18_direct_change_encoding';out=A/'direct_change_encoding';out.mkdir(exist_ok=True)
repo=Path('/home/loganriggs/Coding/reverse_engineer/tensor_language')
for p in [repo,repo/'basis_aligned/polynomial_causal',repo/'basis_aligned/bilinear_quotient/ops']:sys.path.insert(0,str(p))
import bilin18_observed_model_facade as facade
F=torch.nn.functional;torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');seed=10331
panel=json.loads((A/'training_coverage_selector_comparison/panel.json').read_text())['evaluation'];sources={'development':A/'multistream_sparse_baselines/development_states.pt','historical':A/'selector_task_holdouts/states.pt','weekday':A/'reserved_task_format_screen/states.pt'};loaded={k:torch.load(p,weights_only=False) for k,p in sources.items()};ids=sorted({i for p in panel for i in p['indices']});states={};native={};examples={}
for i in ids:
 prefix,index=i.split(':');index=int(index);data=loaded[prefix];states[i]=tuple(v.cuda() for v in data['states'][index]);native[i]=data['native'][index].cuda();examples[i]=data['examples'][index]
geometry=A/'training_coverage_selector_comparison/expanded_pca.pt';checkpointroot=A/'expanded_selector_exposure_control';definition=R/'scripts/run_bilin18_finite_supervised_selectors.py';files=list(sources.values())+[geometry,definition]+[checkpointroot/f'expanded_{k}.pt' for k in ['scalar','block','shared']];hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files};pca=torch.load(geometry,weights_only=False);B=[b.cuda() for b in pca['bases']];mu=pca['center'].cuda();scale=pca['scale'].cuda()
config=dict(experiment='H1044',fit_rows=0,source_hashes=hashes,arms='Three trained endpoint-difference encoders; three direct-change encoders using transferred trained importance; three direct-change encoders with zero importance; dense128perport change reference.',direct='Compute delta=donor-recipient, coefficients=(delta/scale)@B, select using squared delta coefficients, decode once with no center. Add decoded delta to native recipient.',budget='Direct max16perport/48total; endpoint encoders max16perendpoint/port but difference union can be32perport/96total. Dense128perport/384total is capacity reference.',limits='Paired edit operator receives both endpoints. Not autonomous state reconstruction, nor new-task discovery. Trained importance was optimized for endpoints, not direct changes. Fixed random group memberships; no causal output used for support.',evaluation='Same opened H1042/H1043 panels. No selection or fitting on them.')
(R/f'configs/{S}.json').write_text(json.dumps(config,indent=2)+'\n')
tree=ast.parse(definition.read_text());nodes=[n for n in tree.body if (isinstance(n,ast.ClassDef) and n.name=='Selector') or (isinstance(n,ast.FunctionDef) and n.name in ['suffix','KL'])];exec(compile(ast.Module(body=nodes,type_ignores=[]),'<H1033-definitions>','exec'),globals());counts={'suffix':0}
model,_=facade.load_bilin18(snapshot=Path('/home/loganriggs/.cache/huggingface/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240'),device='cuda',dtype=torch.float32,verify_weights_sha256=True)
@torch.no_grad()
def direct(delta,selector=None):
 parts=[];supports=[]
 for k in range(3):
  c=(delta[k].reshape(-1,1152)/scale[k])@B[k]
  if selector is None:support=torch.ones_like(c)
  else:
   scores=c.square()*selector.importance[k].clamp(-2,2).exp()
   if selector.kind=='scalar':support=torch.zeros_like(c).scatter_(1,scores.topk(16,1).indices,1)
   else:
    M=selector.membership[k];budget=4 if selector.kind=='block' else 2;gscore=scores@M.T/(4 if selector.kind=='block' else 8);g=torch.zeros_like(gscore).scatter_(1,gscore.topk(budget,1).indices,1);support=(g@M>0).to(c)
  parts.append(((c*support)@B[k].T*scale[k]).reshape_as(delta[k]));supports.append(support)
 return tuple(parts),supports
records=[];samples=[];endpoint_union_max=0;direct_max=0;antisym=0.;zeroerr=0.;replay=0.;old=json.loads((A/'bilin18_expanded_selector_exposure_control.json').read_text());arms=[('endpoint',k,'trained') for k in ['scalar','block','shared']]+[('direct',k,w) for k in ['scalar','block','shared'] for w in ['untrained','trained']]+[('direct','dense','none')]
with torch.no_grad():
 for mode,kind,weights in arms:
  arm='/'.join([mode,kind,weights]);selector=None
  if kind!='dense':
   selector=Selector(kind)
   if weights=='trained':selector.load_state_dict(torch.load(checkpointroot/f'expanded_{kind}.pt',weights_only=False)['state'])
  pred={};supp={}
  if mode=='endpoint':
   for i in ids:pred[i],supp[i]=selector(states[i])
  first={}
  for p in panel:first.setdefault(p['panel'],p['indices'][0])
  for p in panel:
   for i,d in [p['indices'],p['indices'][::-1]]:
    delta=tuple(states[d][k]-states[i][k] for k in range(3))
    if mode=='endpoint':
     change=tuple(pred[d][k]-pred[i][k] for k in range(3));active=torch.stack([((supp[i][k]+supp[d][k])>0).sum(1) for k in range(3)]).sum(0);endpoint_union_max=max(endpoint_union_max,float(active.max()))
     # Preserve original float32 operation ordering for exact checkpoint metric replay.
     patch=tuple(states[i][k]+pred[d][k]-pred[i][k] for k in range(3))
    else:
     change,ss=direct(delta,selector);active=torch.stack([s.sum(1) for s in ss]).sum(0)
     if kind!='dense':direct_max=max(direct_max,float(active.max()))
     patch=tuple(states[i][k]+change[k] for k in range(3))
     if i==panel[0]['indices'][0]:
      reverse,_=direct(tuple(-v for v in delta),selector);zero,_=direct(tuple(torch.zeros_like(v) for v in delta),selector);antisym=max(antisym,max(float((v+w).abs().max()) for v,w in zip(change,reverse)));zeroerr=max(zeroerr,max(float(v.abs().max()) for v in zero))
    y=suffix(patch);base=native[i];don=native[d];value=float(KL(don.double(),y.double()));den=float(KL(don.double(),base.double()));g=examples[i]['groups'];donor_variant=examples[d]['variant'];errors=[]
    for k in range(3):
     norm=float((delta[k]/scale[k]).square().sum());errors.append(float(((change[k]-delta[k])/scale[k]).square().sum())/norm if norm>0 else None)
    records.append(dict(arm=arm,panel=p['panel'],family=p['family'],row=i,donor=d,paired_KL=value,paired_denominator=den,donor_top1=int(y.argmax()) in g[donor_variant],change_relative_MSE=errors,max_active=float(active.max())))
    if mode=='endpoint':
     prior=next(r for r in old['records'] if r['kind']==kind and r['row']==i);replay=max(replay,abs(value-prior['paired_KL']))
    if i==first[p['panel']]:samples.append(dict(arm=arm,row=i,base=base.cpu(),donor=don.cpu(),paired=y.double().cpu()))
  print('EVALUATED',arm,flush=True)
summary=[]
for key in sorted({(r['arm'],r['panel'],r['family']) for r in records}):
 rr=[r for r in records if (r['arm'],r['panel'],r['family'])==key];summary.append(dict(zip(['arm','panel','family'],key))|dict(paired_relative_KL=sum(r['paired_KL'] for r in rr)/sum(r['paired_denominator'] for r in rr),donor_top1=float(np.mean([r['donor_top1'] for r in rr])),change_relative_MSE=[float(np.mean([r['change_relative_MSE'][k] for r in rr if r['change_relative_MSE'][k] is not None])) for k in range(3)]))
audit=dict(source_hashes_unchanged=all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p)] for p in files),endpoint_checkpoint_KL_replay_max=replay,endpoint_union_max=endpoint_union_max,direct_sparse_max=direct_max,direct_antisymmetry_max=antisym,direct_zero_change_max=zeroerr,suffix_sequences=counts['suffix'],fit_rows=0,finite=all(np.isfinite(r['paired_KL']) for r in records));assert audit['source_hashes_unchanged'] and replay<1e-8 and direct_max<=48 and endpoint_union_max<=96 and antisym<1e-6 and zeroerr==0 and audit['finite']
torch.save(samples,out/'audit_logits.pt');(A/f'{S}.json').write_text(json.dumps(dict(config=config,summary=summary,records=records,audits=audit),indent=2)+'\n');print(json.dumps(audit,indent=2))

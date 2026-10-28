"""H686: same-token/position contextual block swaps with subspace controls."""
import json,time,hashlib
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn.functional as F
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_reader_operation_format import run
from run_bilin18_early_bsf_extended import BSF
STEM='bilin18_bsf_context_swap'
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time();paths={'data':ART/'bilin18_early_bsf_data_captures.pt','atlas':ART/'bilin18_bsf_lexical_audit.json','trainer':ROOT/'scripts/run_bilin18_early_bsf_extended.py','runner':Path(__file__)};hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()};data=torch.load(paths['data'],weights_only=False);atlas=json.loads(paths['atlas'].read_text());cfg=dict(experiment='H686',hashes=hashes,design='All16 H685 residual0/1 candidates. Up to4 deterministic natural example pairs/block, same token and sampled position, different documents, both active. Both directions. Replace decoded block contribution only, versus same-norm global training PCA16 and fixed random16 projections of the full donor-receiver delta; full residual swap reference. Donor-distribution KL gain and receiver KL kept separate.',limitations='Exploratory validation reuse, small pairs/block, partial activation swaps may be off manifold. Full residual swap not an upper bound; donor prefix differs. PCA/random controls match edit size, not dictionary capacity or nonlinear gating.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));pairs=[];coverage=[];checkpoint_hashes={};geometry=[];ids=data['token_ids'];trids=ids[:2048].flatten().cuda();vaids=ids[2048:2304].flatten();states={};patchsets=[]
 for site in ['value0','residual0','residual1']:
  p=ART/'early_bsf_extended'/f'{site}_r16_seed681.pt';checkpoint_hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();ck=torch.load(p,weights_only=False);m=BSF(1152,16).cuda();m.load_state_dict(ck['state_dict']);chosen=[r['block'] for r in atlas['records'] if r['site']==site];raw=data['activations'][site][:2304].reshape(-1,1152).cuda();x=(raw-ck['mean'].cuda())/ck['scale'].cuda();zs=[];ms=[]
  for batch in x.split(512):z,mask=m.encode(batch);zs.append(z[:,chosen]);ms.append(mask[:,chosen])
  z=torch.cat(zs);mask=torch.cat(ms);trainN=32768;valraw=raw[trainN:];D=m.decoder.reshape(144,16,1152)
  if site!='value0':
   centered=raw[:trainN]-raw[:trainN].mean(0);_,eig=torch.linalg.eigh(centered.T@centered/trainN);pca=eig[:,-16:];gen=torch.Generator(device='cuda').manual_seed(686+int(site[-1]));random=torch.linalg.qr(torch.randn(1152,16,generator=gen,device='cuda')).Q;states[site]=valraw.cpu()
  for b,block in enumerate(chosen):
   active=mask[:trainN,b].bool();zz=z[:trainN,b][active]@D[block];ii=trids[active];unique,inv,ct=torch.unique(ii,return_inverse=True,return_counts=True);keep=ct[inv]>=8;unit=F.normalize(zz,dim=-1);means=torch.zeros(len(unique),1152,device='cuda').index_add_(0,inv,unit)/ct[:,None];u=unit[keep];fraction=float((u-means[inv[keep]]).square().sum()/(u-u.mean(0)).square().sum().clamp_min(1e-12));geometry.append(dict(site=site,block=block,decoded_unit_within_token_fraction=fraction,eligible_examples=int(keep.sum())))
   if site=='value0':continue
   activeval=mask[trainN:,b].bool().cpu();groups=defaultdict(list)
   for k in activeval.nonzero().flatten().tolist():groups[int(vaids[k]),k%16].append(k)
   candidates=[]
   for key,group in groups.items():
    group=sorted(group,key=lambda k:hashlib.sha256(f'H686:{site}:{block}:{k}'.encode()).hexdigest())
    for a,c in zip(group[::2],group[1::2]):
     assert a//16!=c//16;candidates.append((a,c))
   candidates=sorted(candidates,key=lambda ac:hashlib.sha256(f'{site}:{block}:{ac}'.encode()).hexdigest())[:4];coverage.append(dict(site=site,block=block,pairs=len(candidates)))
   for a,c in candidates:
    for receiver,donor in [(a,c),(c,a)]:
     delta=valraw[donor]-valraw[receiver];learned=(z[trainN+donor,b]-z[trainN+receiver,b])@D[block]*ck['scale'].cuda();norm=learned.norm();pv=(delta@pca)@pca.T;rv=(delta@random)@random.T;assert pv.norm()>0 and rv.norm()>0;patchsets.append(dict(block=learned.cpu(),pca=pv.cpu()*(norm/pv.norm()).cpu(),random=rv.cpu()*(norm/rv.norm()).cpu(),full=delta.cpu()));pairs.append(dict(site=site,block=block,receiver=receiver,donor=donor,position=data['positions'][receiver%16],token_id=int(vaids[receiver]),norm=float(norm)))
 del m,raw,x,z,mask;model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 endpoints=sorted({p[k] for p in pairs for k in ['receiver','donor']});native={};replay=[];forwards=0
 for endpoint in endpoints:
  doc,pos=endpoint//16,data['positions'][endpoint%16];hooks=[];capt={}
  for layer in [0,1]:
   def get(m,args,out,layer=layer):capt[layer]=out[0][0,-1].cpu()
   hooks.append(model.blocks[layer].register_forward_hook(get))
  try:logits,_=run(model,data['all_token_ids'][2048+doc:2048+doc+1,:pos+1].cuda());native[endpoint]=logits[0].cpu()
  finally:
   for h in hooks:h.remove()
  for layer in [0,1]:truth=states[f'residual{layer}'][endpoint];replay.append(float((capt[layer]-truth).norm()/truth.norm()))
  forwards+=1
 records=[];outputs={};buckets=defaultdict(list)
 for k,p in enumerate(pairs):buckets[p['site'],p['position']].append(k)
 for mode in ['block','pca','random','full']:
  for (site,pos),inds in buckets.items():
   for st in range(0,len(inds),8):
    bb=inds[st:st+8];patch=torch.stack([patchsets[k][mode] for k in bb]);layer=int(site[-1])
    def edit(m,args,out):
     x,bus=out;x=x.clone();x[:,-1]+=patch.to(x);return x,bus
    hook=model.blocks[layer].register_forward_hook(edit)
    try:logits,_=run(model,torch.stack([data['all_token_ids'][2048+pairs[k]['receiver']//16,:pos+1] for k in bb]).cuda())
    finally:hook.remove()
    forwards+=len(bb)
    for n,k in enumerate(bb):
     p=pairs[k];base=native[p['receiver']].double().log_softmax(-1);don=native[p['donor']].double().log_softmax(-1);edited=logits[n].double().log_softmax(-1).cpu();distance=float((don.exp()*(don-base)).sum());newdistance=float((don.exp()*(don-edited)).sum());records.append(dict(**p,mode=mode,donor_kl_baseline=distance,donor_kl_edited=newdistance,donor_kl_gain=distance-newdistance,receiver_kl=float((base.exp()*(base-edited)).sum())))
  print('DONE',mode,flush=True)
 audits=dict(hashes_unchanged={k:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[k] for k,p in paths.items()},checkpoints_unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in checkpoint_hashes.items()),whole_model_forwards=forwards,native_endpoints=len(endpoints),directed_pairs=len(pairs),prefix_state_replay_max_relative=max(replay),test_evaluations=0,finite=all(torch.isfinite(torch.tensor(r['donor_kl_gain'])).item() for r in records));assert max(replay)<1e-4;(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,geometry=geometry,coverage=coverage,checkpoint_hashes=checkpoint_hashes,audits=audits,elapsed_seconds=time.time()-start),indent=2));torch.save(dict(pairs=pairs,patches=patchsets,native_logits=native),ART/f'{STEM}_captures.pt');print(audits)
if __name__=='__main__':main()

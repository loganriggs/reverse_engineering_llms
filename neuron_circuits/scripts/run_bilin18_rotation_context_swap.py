"""H708: natural same-token/position swaps of stable rotation coordinates."""
import torch,json,time,hashlib
from pathlib import Path
from collections import defaultdict
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_reader_operation_format import run
STEM='bilin18_rotation_context_swap'
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time();paths={'data':ART/'bilin18_early_bsf_data_captures.pt','spectra':ART/'bilin18_early_rotation_spectra_captures.pt','rotations':ART/'bilin18_early_sparse_rotations_captures.pt','atlas':ART/'bilin18_rotation_stability_context.json','model':SNAPSHOT/'pytorch_model.bin','loader':ROOT/'scripts/run_bilin18_qkov_causal_edges.py','forward':ROOT/'scripts/run_bilin18_reader_operation_format.py','runner':Path(__file__)};hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()};data=torch.load(paths['data'],weights_only=False);sp=torch.load(paths['spectra'],weights_only=False)['states']['residual1'];Q=torch.load(paths['rotations'],weights_only=False)['rotations']['residual1','learned',7061].cuda();atlas=json.loads(paths['atlas'].read_text());chosen=[r['direction'] for r in atlas['lexical'] if r['site']=='residual1'];mu=sp['mean'].cuda().float();V=sp['eigenvectors'][:,:256].cuda().float();sd=sp['eigenvalues'][:256].cuda().float().sqrt();raw=data['activations']['residual1'][2048:2304].reshape(-1,1152).cuda();D=(V*sd)@Q;norms=D.norm(dim=0);D=D/norms;code=(((raw-mu)@V)/sd)@Q*norms;ids=code.abs().topk(32,dim=-1).indices;mask=torch.zeros_like(code).scatter_(1,ids,1);tokens=data['token_ids'][2048:2304].flatten();pca=V[:,0];gen=torch.Generator(device='cuda').manual_seed(708);random=torch.nn.functional.normalize(torch.randn(1152,generator=gen,device='cuda'),dim=0);pairs=[];patches=[];coverage=[]
 for f in chosen:
  groups=defaultdict(list)
  for k in (mask[:,f]>0).nonzero().flatten().cpu().tolist():groups[int(tokens[k]),k%16].append(k)
  candidates=[]
  for group in groups.values():
   group=sorted(group,key=lambda k:hashlib.sha256(f'H708:{f}:{k}'.encode()).hexdigest())
   for a,b in zip(group[::2],group[1::2]):assert a//16!=b//16;candidates.append((a,b))
  candidates=sorted(candidates,key=lambda ab:hashlib.sha256(f'H708:{f}:{ab}'.encode()).hexdigest())[:4];coverage.append(dict(direction=f,pairs=len(candidates)))
  for a,b in candidates:
   for receiver,donor in [(a,b),(b,a)]:
    delta=raw[donor]-raw[receiver];component=(code[donor,f]-code[receiver,f])*D[:,f];n=component.norm();projected=(delta@D[:,f])*D[:,f];pv=(delta@pca)*pca;rv=(delta@random)*random;assert pv.norm()>0 and rv.norm()>0;patches.append(dict(coordinate=component.cpu(),decoder_projection=projected.cpu(),pca=pv.cpu()*float(n/pv.norm()),random=rv.cpu()*float(n/rv.norm()),full=delta.cpu()));pairs.append(dict(direction=f,receiver=receiver,donor=donor,position=int(data['positions'][receiver%16]),token_id=int(tokens[receiver]),coordinate_norm=float(n),projection_norm=float(projected.norm()),coordinate_projection_same_sign=bool((component@projected)>=0)))
 cfg=dict(experiment='H708',hashes=hashes,design='Eight H707stable residual1directions; up to4hash-selected same-token/position pairs each, differentdocs, bothTop32active, bothdirections. Swap single dual-coordinate decoded contribution at block1output. Compare rank1globalPCA/random projected deltas rescaled to same norm, unscaled Euclidean decoder projection, fullresidualswap. Measure donor KL gain and receiver KL separately.',limitations='Small exploratory validation panel; no correctness target. Fullswap notupperbound. Projection is deliberately unscaled and notnormmatched; oblique coordinate substitutions differ from Euclidean projections. One fixedrandomdirection, no randomensemble.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 endpoints=sorted({p[k] for p in pairs for k in ['receiver','donor']});native={};replay=[];forwards=0
 for endpoint in endpoints:
  doc,pos=endpoint//16,int(data['positions'][endpoint%16]);capt={}
  def get(m,args,out):capt['x']=out[0][0,-1].cpu()
  h=model.blocks[1].register_forward_hook(get)
  try:logits,_=run(model,data['all_token_ids'][2048+doc:2048+doc+1,:pos+1].cuda());native[endpoint]=logits[0].cpu()
  finally:h.remove()
  replay.append(float((capt['x']-raw[endpoint].cpu()).norm()/raw[endpoint].norm().cpu()));forwards+=1
 assert max(replay)<1e-4;records=[];buckets=defaultdict(list)
 for k,p in enumerate(pairs):buckets[p['position']].append(k)
 for mode in ['coordinate','decoder_projection','pca','random','full']:
  for pos,inds in buckets.items():
   for st in range(0,len(inds),8):
    bb=inds[st:st+8];patch=torch.stack([patches[k][mode] for k in bb])
    def edit(m,args,out):
     x,bus=out;x=x.clone();x[:,-1]+=patch.to(x);return x,bus
    hook=model.blocks[1].register_forward_hook(edit)
    try:logits,_=run(model,torch.stack([data['all_token_ids'][2048+pairs[k]['receiver']//16,:pos+1] for k in bb]).cuda())
    finally:hook.remove()
    forwards+=len(bb)
    for n,k in enumerate(bb):
     p=pairs[k];base=native[p['receiver']].double().log_softmax(-1);don=native[p['donor']].double().log_softmax(-1);edited=logits[n].double().log_softmax(-1).cpu();distance=float((don.exp()*(don-base)).sum());newdistance=float((don.exp()*(don-edited)).sum());records.append(dict(**p,mode=mode,edit_norm=float(patch[n].norm()),donor_kl_baseline=distance,donor_kl_edited=newdistance,donor_kl_gain=distance-newdistance,receiver_kl=float((base.exp()*(base-edited)).sum())))
  (ART/f'{STEM}.partial.json').write_text(json.dumps(dict(status='running',records=records,coverage=coverage),indent=2));print('DONE',mode,flush=True)
 audits=dict(hashes_unchanged={k:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[k] for k,p in paths.items()},finite=all(torch.isfinite(torch.tensor(r['donor_kl_gain'])).item() for r in records),directed_pairs=len(pairs),native_endpoints=len(endpoints),prefix_state_replay_max_relative=max(replay),whole_model_forwards=forwards,test_evaluations=0);assert all(audits['hashes_unchanged'].values()) and audits['finite'];torch.save(dict(pairs=pairs,patches=patches,native_logits=native),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,coverage=coverage,audits=audits,elapsed_seconds=time.time()-start),indent=2));print(audits,flush=True)
if __name__=='__main__':main()

"""H837 frozen sparse/grouped featurizers on a new document-disjoint panel."""
import json,time,hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_balanced_shared_featurizers import Sparse,BalancedShared
from run_bilin18_early_local_geometry import sha
S='bilin18_frozen_featurizer_confirmation'
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time();paths={'old_manifest':ART/'bilin18_early_bsf_data_manifest.json','h824_manifest':ART/'bilin18_fresh_early_geometry_panel_manifest.json','fits':ART/'bilin18_balanced_shared_featurizers_captures.pt','global':ART/'bilin18_operator_state_prediction_captures.pt','weighted':ART/'bilin18_context_operator_bases_fits.pt','runner':Path(__file__),'sparse_implementation':ROOT/'scripts/run_bilin18_balanced_shared_featurizers.py','shared_implementation':ROOT/'scripts/feature_structures/models.py','model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};old=json.loads(paths['old_manifest'].read_text());prior=json.loads(paths['h824_manifest'].read_text());exclude={r['document_hash'] for r in old['records']+prior['records']};windows={r['window_sha256_int64'] for r in old['records']};arrays={}
 def tokens(r):
  if r['source'] not in arrays:arrays[r['source']]=np.load(r['source'],mmap_mode='r')
  return np.asarray(arrays[r['source']][r['source_row'],:128],dtype=np.int64)
 for r in prior['records']:windows.add(hashlib.sha256(tokens(r).tobytes()).hexdigest())
 pool={}
 for name,h in old['source_hashes'].items():
  if not name.endswith('.manifest.json'):continue
  p=Path(name);assert sha(p)==h;arr=p.with_name(p.name.replace('.manifest.json','.npy'));assert sha(arr)==old['source_hashes'][str(arr)]
  for i,r in enumerate(json.loads(p.read_text())['records']):
   dh=r['document_token_sha256']
   if dh not in exclude:pool.setdefault(dh,{'document_hash':dh,'source':str(arr),'source_row':i})
 selected=[];ids=[];newwindows=set()
 for dh in sorted(pool,key=lambda h:hashlib.sha256(('H837:'+h).encode()).hexdigest()):
  r=pool[dh];t=tokens(r);wh=hashlib.sha256(t.tobytes()).hexdigest()
  if wh in windows or wh in newwindows:continue
  newwindows.add(wh);selected.append({**r,'window_sha256_int64':wh});ids.append(t)
  if len(ids)==512:break
 assert len(ids)==512;ids=torch.tensor(np.stack(ids));positions=[7+8*(int(hashlib.sha256(('H837pos:'+r['document_hash']).encode()).hexdigest(),16)%16) for r in selected]
 cfg={'experiment':'H837','hashes':hashes,'selection':'512 H837 hash-selected documents from existing corpus pool, exclude all2560 H681 plus1024 H824 document hashes and128-token window hashes. One deterministic sampled position each.','fit':'None; all20 H836 checkpoints frozen, both seeds, plus rank64 global/residual/weighted/routed-weighted H832/H833 baselines.','probes':'Eight new orthonormal Gaussian raw directions seed837+layer, plus recipient native attention-output direction normalized to unit norm. The latter is a state-dependent write direction, not a donor swap or validated task direction.','downstream_lock':{'document_indices':list(range(32)),'probe_indices':[0,8],'signed_doses':[-.1,.1],'dose_units':'recipient preMLP RMS times unit direction','reference':'natural finite MLP response; compare approximate derivative responses and zero MLP response; preserve skip perturbation'},'scope':'New to early-featurizer panels; source pool reused research documents. No new model-training holdout or corpus acquisition. Confirmation selection includes all token identities, seen/unseen retained.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');(ART/f'{S}_manifest.json').write_text(json.dumps({'records':selected,'positions':positions,'source_hashes':old['source_hashes']},indent=2)+'\n');glob=torch.load(paths['global'],weights_only=False);sparse=torch.load(paths['fits'],weights_only=False)['states'];weighted=torch.load(paths['weighted'],weights_only=False)['fits'];model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 caps={l:{k:[] for k in ['premlp','mlpout','attention']} for l in [1,2]}
 for st in range(0,512,16):
  batch=ids[st:st+16].cuda();x0=F.rms_norm(model.embedding(batch),(1152,));x=x0;bus=None;ix=torch.arange(len(batch),device='cuda');pi=torch.tensor(positions[st:st+len(batch)],device='cuda')
  for l in range(3):
   block=model.blocks[l];raw=block.lambdas[0]*x+block.lambdas[1]*x0;att,bus=block.attn(F.rms_norm(raw,(1152,)),bus);pre=raw+att;mlpout=block.mlp(F.rms_norm(pre,(1152,)));x=pre+mlpout
   if l in caps:
    for key,t in [('premlp',pre),('mlpout',mlpout),('attention',att)]:caps[l][key].append(t[ix,pi].cpu())
 for l in caps:
  for k in caps[l]:caps[l][k]=torch.cat(caps[l][k])
 print('CAPTURED',flush=True);eps=torch.finfo(torch.float32).eps;tids=ids[torch.arange(512),torch.tensor(positions)];predictions={};probes={};records=[];unionerr=[];names=[]
 for l in [1,2]:
  f=glob['fits'][l];b=caps[l]['premlp'].double().cuda();rad=(b.square().mean(1,keepdim=True)+eps).sqrt();u=b/rad;lookup={int(t):i for i,t in enumerate(f['token_ids'])};seen=torch.tensor([int(t) in lookup for t in tids],device='cuda');means=f['token_shape_mean'].cuda();mu=f['shape_mean'].cuda();tm=torch.stack([means[lookup[int(t)]] if int(t) in lookup else mu for t in tids]);res=u-tm
  def raw(shape):return shape/shape.square().mean(1,keepdim=True).sqrt().clamp_min(1e-30)*(rad.square()-eps).sqrt()
  B=f['basis'].cuda();P=weighted[l]['residual_pca'].cuda();E=weighted[l]['weighted_encoder'].cuda();D=weighted[l]['weighted_decoder'].cuda();pred={'global64':raw(mu+(u-mu)@B@B.T),'residual64':raw(tm+res@P@P.T),'weighted64':raw(tm+res@E@D.T)};pred['routed_weighted64']=torch.where(seen[:,None],pred['weighted64'],pred['global64'])
  for kind in ['scalar48','scalar64','block8','shared_fixed','shared_learned']:
   for seed in [8351,8352]:
    ck=sparse[l,kind,seed];m=BalancedShared(seed,kind=='shared_fixed').cuda() if kind.startswith('shared') else Sparse(8 if kind=='block8' else 1,48 if kind=='scalar48' else 64).cuda();m.load_state_dict(ck['state_dict']);outs=[m(v) for v in (res/ck['scale']).float().split(128)];z=torch.cat([o['reconstruction'] for o in outs]).double();pred[f'{kind}_{seed}']=raw(tm+z*ck['scale'])
    if kind.startswith('shared'):
     for o in outs:unionerr.append(float((o['support']-(o['gates'].detach()@o['membership'].detach()>0).float()).abs().max()))
    del m,outs
  gen=torch.Generator(device='cuda').manual_seed(837+l);random=torch.linalg.qr(torch.randn(1152,8,dtype=torch.float64,device='cuda',generator=gen)).Q.T;natural=caps[l]['attention'].double().cuda();assert bool((natural.norm(dim=1)>0).all());natural=natural/natural.norm(dim=1,keepdim=True);directions=torch.cat([random[:,None,:].expand(8,512,1152),natural[None]],dim=0);probes[l]=directions.cpu();mlp=model.blocks[l].mlp;assert not mlp.config.gated;L,R,D=[getattr(mlp,n).weight.double() for n in ['Left','Right','Down']]
  def response(base,v):
   r=(base.square().mean(1,keepdim=True)+eps).sqrt();s=base/r;q=v/r-base*(base*v).mean(1,keepdim=True)/r.pow(3);return F.linear(F.linear(s,L)*F.linear(q,R)+F.linear(q,L)*F.linear(s,R),D)
  for di,v in enumerate(directions):
   truth=response(b,v);den=truth.square().sum(1)
   for name,p in pred.items():
    y=response(p,v);err=(y-truth).square().sum(1)
    for i in range(512):records.append({'layer':l,'document':i,'seen':bool(seen[i]),'probe':di,'model':name,'squared_error':float(err[i]),'truth_squared_norm':float(den[i])})
  for name,p in pred.items():predictions[l,name]=p.cpu()
  names=list(pred);print('EVAL',l,flush=True)
 summary=[]
 for l in [1,2]:
  for kind in ['random','attention_write']:
   for stratum in ['all','seen','unseen']:
    for name in names:
     rr=[r for r in records if r['layer']==l and r['model']==name and (r['probe']==8)==(kind=='attention_write') and (stratum=='all' or r['seen']==(stratum=='seen'))];summary.append({'layer':l,'probes':kind,'stratum':stratum,'model':name,'documents':len({r['document'] for r in rr}),'relative_RMSE':float(np.sqrt(sum(r['squared_error'] for r in rr)/sum(r['truth_squared_norm'] for r in rr)))})
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'document_disjoint':not bool(exclude&{r['document_hash'] for r in selected}),'window_disjoint':not bool(windows&newwindows),'union_exact_max_abs':max(unionerr),'three_block_prefix_forwards':512,'whole_model_forwards':0,'test_evaluations':0,'finite':all(np.isfinite(r['relative_RMSE']) for r in summary)};assert audits['hashes_unchanged'] and audits['document_disjoint'] and audits['window_disjoint'] and max(unionerr)==0 and audits['finite'];torch.save({'captures':caps,'predicted_states':predictions,'directions':probes,'all_token_ids':ids,'positions':positions,'token_ids':tids},ART/f'{S}_captures.pt');(ART/f'{S}.json').write_text(json.dumps({'config':cfg,'records':records,'summary':summary,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(audits,flush=True)
if __name__=='__main__':main()

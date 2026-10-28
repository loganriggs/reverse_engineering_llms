"""H834 frozen operator-state representations on unused documents and probes."""
import json,time,hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_early_local_geometry import sha
S='bilin18_operator_frozen_confirmation'
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time();paths={'global_fits':ART/'bilin18_operator_state_prediction_captures.pt','context_fits':ART/'bilin18_context_operator_bases_fits.pt','panel':ART/'bilin18_fresh_early_geometry_panel_captures.pt','h825':ART/'bilin18_fresh_geometry_causal_captures.pt','h827':ART/'bilin18_fixed_direction_recipients_captures.pt','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};glob=torch.load(paths['global_fits'],weights_only=False);ctx=torch.load(paths['context_fits'],weights_only=False);panel=torch.load(paths['panel'],weights_only=False);a=torch.load(paths['h825'],weights_only=False);b=torch.load(paths['h827'],weights_only=False);exclude={e//16 for e in a['native']}|{e//16 for g in b['groups'] for e in g['recipients']}|set(glob['validation_documents']);chosen=[i for i in range(1024) if i not in exclude];chosen.sort(key=lambda i:hashlib.sha256(f'H834:{i}'.encode()).hexdigest());assert len(chosen)>128;vp=[int(hashlib.sha256(f'H834pos:{i}'.encode()).hexdigest(),16)%16 for i in chosen];positions=panel['positions'];ids=panel['all_token_ids'][chosen].clone();del panel,a,b
 names=[f'{kind}{rank}' for rank in [16,64] for kind in ['global','residual','weighted','routed_residual','routed_weighted']];cfg={'experiment':'H834','hashes':hashes,'documents':chosen,'position_indices':vp,'fit':'None. Frozen H832/H833 fits, means, encoders/decoders and ranks16/64.','routing':'Seen among original sampled training token ids: token residual model; unseen: frozen global PCA same rank. No threshold tuning.','probe_seed':'834+layer,8 orthonormal probes independent of fitting/development probes.','metrics':'Local RMS-plus-bilinear derivative RMSE and cosine; seen/unseen and aggregate; true radius explicitly available to every model.','scope':'Fresh confirmation relative to prior causal/response evaluations within H824 panel, not new corpus or original reserved test. All remaining unused documents included.','models':names};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 caps={l:[] for l in [1,2]}
 for st in range(0,len(ids),16):
  batch=ids[st:st+16].cuda();x0=F.rms_norm(model.embedding(batch),(1152,));x=x0;bus=None
  for l in range(3):
   block=model.blocks[l];raw=block.lambdas[0]*x+block.lambdas[1]*x0;attn,bus=block.attn(F.rms_norm(raw,(1152,)),bus);pre=raw+attn
   if l in caps:caps[l].append(pre[torch.arange(len(batch),device='cuda'),torch.tensor([positions[k] for k in vp[st:st+len(batch)]],device='cuda')].cpu())
   x=pre+block.mlp(F.rms_norm(pre,(1152,)))
 for l in caps:caps[l]=torch.cat(caps[l])
 vids=ids[torch.arange(len(ids)),torch.tensor([positions[k] for k in vp])];eps=torch.finfo(torch.float32).eps;records=[];probeaudit=[];routeaudit=[];probes={}
 for l in [1,2]:
  f=glob['fits'][l];c=ctx['fits'][l];base=caps[l].double().cuda();rad=(base.square().mean(1,keepdim=True)+eps).sqrt();u=base/rad;mu=f['shape_mean'].cuda();lookup={int(t):i for i,t in enumerate(f['token_ids'])};seen=torch.tensor([int(t) in lookup for t in vids],device='cuda');tmean=f['token_shape_mean'].cuda();tm=torch.stack([tmean[lookup[int(t)]] if int(t) in lookup else mu for t in vids]);pred={}
  def raw(s):return s/s.square().mean(1,keepdim=True).sqrt().clamp_min(1e-30)*(rad.square()-eps).sqrt()
  for rank in [16,64]:
   B=f['basis'][:,:rank].cuda();P=c['residual_pca'][:,:rank].cuda();E=c['weighted_encoder'][:,:rank].cuda();D=c['weighted_decoder'][:,:rank].cuda();pred[f'global{rank}']=raw(mu+(u-mu)@B@B.T);pred[f'residual{rank}']=raw(tm+(u-tm)@P@P.T);pred[f'weighted{rank}']=raw(tm+(u-tm)@E@D.T)
   for kind in ['residual','weighted']:
    pred[f'routed_{kind}{rank}']=torch.where(seen[:,None],pred[f'{kind}{rank}'],pred[f'global{rank}']);routeaudit.append(float((pred[f'routed_{kind}{rank}'][~seen]-pred[f'global{rank}'][~seen]).abs().max()) if (~seen).any() else 0.)
  gen=torch.Generator(device='cuda').manual_seed(834+l);dirs=torch.linalg.qr(torch.randn(1152,8,device='cuda',dtype=torch.float64,generator=gen)).Q.T;probes[l]=dirs.cpu();probeaudit.append(float((dirs@dirs.T-torch.eye(8,device='cuda')).abs().max()));mlp=model.blocks[l].mlp;assert not mlp.config.gated;L,R,D=[getattr(mlp,n).weight.double() for n in ['Left','Right','Down']]
  def response(b,v):
   r=(b.square().mean(1,keepdim=True)+eps).sqrt();s=b/r;q=v/r-b*(b*v).mean(1,keepdim=True)/r.pow(3);return F.linear(F.linear(s,L)*F.linear(q,R)+F.linear(q,L)*F.linear(s,R),D)
  for di,v in enumerate(dirs):
   truth=response(base,v);den=truth.square().sum(1)
   for name,p in pred.items():
    y=response(p,v);err=(y-truth).square().sum(1);co=(y*truth).sum(1)/(y.norm(dim=1)*truth.norm(dim=1)).clamp_min(1e-30)
    for i in range(len(chosen)):records.append({'layer':l,'document':chosen[i],'seen':bool(seen[i]),'probe':di,'model':name,'squared_error':float(err[i]),'truth_squared_norm':float(den[i]),'cosine':float(co[i])})
  print('DONE',l,flush=True)
 summary=[]
 for l in [1,2]:
  for stratum in ['all','seen','unseen']:
   for name in names:
    rr=[r for r in records if r['layer']==l and r['model']==name and (stratum=='all' or r['seen']==(stratum=='seen'))];summary.append({'layer':l,'stratum':stratum,'model':name,'documents':len(rr)//8,'relative_RMSE':float(np.sqrt(sum(r['squared_error'] for r in rr)/sum(r['truth_squared_norm'] for r in rr))),'mean_cosine':float(np.mean([r['cosine'] for r in rr]))})
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'document_exclusion_check':not bool(set(chosen)&exclude),'documents':len(chosen),'routed_unseen_exact_global_error':max(routeaudit),'probe_orthogonality_max_abs':max(probeaudit),'three_block_prefix_forwards':len(chosen),'whole_model_forwards':0,'test_evaluations':0,'finite':all(np.isfinite(r['relative_RMSE']) for r in summary)};assert audits['hashes_unchanged'] and audits['document_exclusion_check'] and max(routeaudit)==0 and audits['finite'];torch.save({'captures':caps,'probe_directions':probes,'documents':chosen,'position_indices':vp,'token_ids':vids},ART/f'{S}_captures.pt');(ART/f'{S}.json').write_text(json.dumps({'config':cfg,'records':records,'summary':summary,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(audits,flush=True)
if __name__=='__main__':main()

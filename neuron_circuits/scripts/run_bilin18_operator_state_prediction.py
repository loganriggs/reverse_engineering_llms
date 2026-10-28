"""H832 training-only low-rank state prediction on unused natural documents."""
import json,time,hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_early_local_geometry import sha
S='bilin18_operator_state_prediction'
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time();paths={'training':ART/'bilin18_early_bsf_data_captures.pt','fresh':ART/'bilin18_fresh_early_geometry_panel_captures.pt','h825':ART/'bilin18_fresh_geometry_causal_captures.pt','h827':ART/'bilin18_fixed_direction_recipients_captures.pt','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};old=torch.load(paths['training'],weights_only=False);fresh=torch.load(paths['fresh'],weights_only=False);a=torch.load(paths['h825'],weights_only=False);b=torch.load(paths['h827'],weights_only=False);exclude={e//16 for e in a['native']}|{e//16 for g in b['groups'] for e in g['recipients']};eligible=[i for i in range(1024) if i not in exclude];eligible.sort(key=lambda i:hashlib.sha256(f'H832:{i}'.encode()).hexdigest());chosen=eligible[:512];assert len(chosen)==512;positions=old['positions'];vp=[int(hashlib.sha256(f'H832pos:{i}'.encode()).hexdigest(),16)%16 for i in chosen];trainids=old['all_token_ids'][:2048].clone();validids=fresh['all_token_ids'][chosen].clone();del old,fresh,a,b
 cfg={'experiment':'H832','hashes':hashes,'train':'Original2048 BSF training documents, all16 sampled positions;32768 states per site.','validation_documents':chosen,'validation_position_indices':vp,'validation':'512 H824 documents excluding ALL H825 native endpoint and H827 recipient documents. One deterministic position per document; no outcomes used for selection. Prior activation capture is not new outcome evaluation.','sites':[1,2],'probes':'Eight fixed orthonormal Gaussian raw input directions, seed832+layer; no fitting on response targets.','models':['global_raw','token_raw','global_shape_radius','token_shape_radius','pca4_radius','pca16_radius','pca64_radius'],'access':'raw fixed baselines use training global/token mean b only. Shape+radius models use exact one-scalar recipient r=sqrt(mean(b^2)+eps). PCA additionally encodes full normalized state into rank-r coordinates, reconstructs and renormalizes shape. All local derivatives computed from reconstructed raw b; no true-state RMS derivative leaked. Token means fall back to global if absent.','limitations':'Natural-state local operator benchmark, not full-model fidelity or semantic/task labels. No BSF trained at these preMLP sites yet. Training data reused, validation new only to this causal benchmark.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 captures={};prefix=0
 for split,ids in [('train',trainids),('validation',validids)]:
  out={l:[] for l in [1,2]}
  for st in range(0,len(ids),16):
   batch=ids[st:st+16].cuda();x0=F.rms_norm(model.embedding(batch),(1152,));x=x0;bus=None
   for l in range(3):
    block=model.blocks[l];raw=block.lambdas[0]*x+block.lambdas[1]*x0;attn,bus=block.attn(F.rms_norm(raw,(1152,)),bus);pre=raw+attn
    if l in out:
     z=pre[:,positions] if split=='train' else pre[torch.arange(len(batch),device='cuda'),torch.tensor([positions[k] for k in vp[st:st+len(batch)]],device='cuda')];out[l].append(z.cpu())
    x=pre+block.mlp(F.rms_norm(pre,(1152,)))
   prefix+=len(batch)
  for l in out:captures[split,l]=torch.cat(out[l]).reshape(-1,1152)
  print('CAPTURE',split,flush=True)
 tids=trainids[:,positions].flatten();vids=validids[torch.arange(512),torch.tensor([positions[k] for k in vp])];eps=torch.finfo(torch.float32).eps;records=[];fits={};checks=[]
 for l in [1,2]:
  x=captures['train',l].double().cuda();val=captures['validation',l].double().cuda();rad=(x.square().mean(1,keepdim=True)+eps).sqrt();u=x/rad;vr=(val.square().mean(1,keepdim=True)+eps).sqrt();vu=val/vr;mu=u.mean(0);mb=x.mean(0);xc=u-mu;_,V=torch.linalg.eigh(xc.T@xc/len(x));B=V[:,-64:].flip(1);unique,inv=torch.unique(tids,return_inverse=True);cnt=torch.bincount(inv).double().cuda();meanb=torch.zeros(len(unique),1152,dtype=torch.float64,device='cuda').index_add_(0,inv.cuda(),x)/cnt[:,None];meanu=torch.zeros_like(meanb).index_add_(0,inv.cuda(),u)/cnt[:,None];lookup={int(t):i for i,t in enumerate(unique)};tb=torch.stack([meanb[lookup[int(t)]] if int(t) in lookup else mb for t in vids]);tu=torch.stack([meanu[lookup[int(t)]] if int(t) in lookup else mu for t in vids]);unseen=sum(int(t) not in lookup for t in vids)
  def shaperaw(shape):return shape/shape.square().mean(1,keepdim=True).sqrt().clamp_min(1e-30)*(vr.square()-eps).clamp_min(0).sqrt()
  pred={'global_raw':mb.expand_as(val),'token_raw':tb,'global_shape_radius':shaperaw(mu.expand_as(val)),'token_shape_radius':shaperaw(tu)}
  for rank in [4,16,64]:bb=B[:,:rank];pred[f'pca{rank}_radius']=shaperaw(mu+(vu-mu)@bb@bb.T)
  gen=torch.Generator(device='cuda').manual_seed(832+l);dirs=torch.linalg.qr(torch.randn(1152,8,dtype=torch.float64,device='cuda',generator=gen)).Q.T;mlp=model.blocks[l].mlp;assert not mlp.config.gated;L,R,D=[getattr(mlp,n).weight.double() for n in ['Left','Right','Down']]
  def response(base,v):
   r=(base.square().mean(1,keepdim=True)+eps).sqrt();s=base/r;q=v/r-base*(base*v).mean(1,keepdim=True)/r.pow(3);return F.linear(F.linear(s,L)*F.linear(q,R)+F.linear(q,L)*F.linear(s,R),D)
  for di,v in enumerate(dirs):
   truth=response(val,v);den=truth.square().sum(1)
   for name,p in pred.items():
    y=response(p,v);err=(y-truth).square().sum(1);align=(y*truth).sum(1)/(y.norm(dim=1)*truth.norm(dim=1)).clamp_min(1e-30)
    for i in range(512):records.append({'layer':l,'document':chosen[i],'probe':di,'model':name,'squared_error':float(err[i]),'truth_squared_norm':float(den[i]),'cosine':float(align[i])})
  fits[l]={'shape_mean':mu.cpu(),'raw_mean':mb.cpu(),'basis':B.cpu(),'token_ids':unique,'token_raw_mean':meanb.cpu(),'token_shape_mean':meanu.cpu(),'directions':dirs.cpu(),'unseen_validation_tokens':unseen};checks.append(float((B.T@B-torch.eye(64,device='cuda')).abs().max()));print('EVAL',l,flush=True)
 summary=[]
 for l in [1,2]:
  for name in cfg['models']:
   rr=[r for r in records if r['layer']==l and r['model']==name];summary.append({'layer':l,'model':name,'relative_RMSE':float(np.sqrt(sum(r['squared_error'] for r in rr)/sum(r['truth_squared_norm'] for r in rr))),'mean_cosine':float(np.mean([r['cosine'] for r in rr]))})
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'unused_document_check':not bool(set(chosen)&exclude),'orthogonality_max_abs':max(checks),'finite':all(np.isfinite(r['relative_RMSE']) for r in summary),'three_block_prefix_forwards':prefix,'whole_model_forwards':0,'test_evaluations':0};assert audits['hashes_unchanged'] and audits['unused_document_check'] and max(checks)<1e-10 and audits['finite'];torch.save({'captures':captures,'fits':fits,'train_token_ids':tids,'validation_token_ids':vids,'validation_documents':chosen,'validation_positions':vp},ART/f'{S}_captures.pt');(ART/f'{S}.json').write_text(json.dumps({'config':cfg,'records':records,'summary':summary,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(json.dumps({'summary':summary,'audits':audits},indent=2),flush=True)
if __name__=='__main__':main()

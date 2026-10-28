"""H833 token-conditioned residual PCA and response-weighted low-rank state bases."""
import json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_early_local_geometry import sha
S='bilin18_context_operator_bases'
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time();paths={'data':ART/'bilin18_operator_state_prediction_captures.pt','previous':ART/'bilin18_operator_state_prediction.json','model':SNAPSHOT/'pytorch_model.bin','runner':Path(__file__)};hashes={k:sha(p) for k,p in paths.items()};cfg={'experiment':'H833','hashes':hashes,'fit':'Training-only normalized-state residual after subtracting training token mean;9016 token means retained. Global mean fallback unseen validation tokens.','models':['residual_pca4','residual_pca16','residual_pca64','response_weighted4','response_weighted16','response_weighted64'],'weighted_objective':'For8 independent orthonormal fit probes seed833+layer, T_v=D[diag(Rv)L+diag(Lv)R]. W=mean(T_v^T T_v)+.001*trace(mean)/1152*I. Minimize weighted residual reconstruction via PCA of W^1/2 residual; encode E=W^1/2 B, decode F=W^-1/2 B. Fit excludes full RMS derivative and final shape normalization, so objective is a surrogate.','evaluation':'Same H832512-document development validation and independent8 probes seed832+layer. Reconstruct token shape plus rank-r residual; normalize shape, restore exact recorded radius, recompute full RMS/bilinear derivative. No true-state derivative access.','budget':'r context coordinates plus one radius scalar and token identity; same token table across candidates. Weighted encoder/decoder differ and store2*1152*r entries versus shared PCA basis1152*r. No overall storage parity claim.','scope':'Post-H832 development comparison; unused confirmation documents and reserved test untouched.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');data=torch.load(paths['data'],weights_only=False);previous=json.loads(paths['previous'].read_text());model=load_full_model('cuda');eps=torch.finfo(torch.float32).eps;records=[];fits={};metricerr=[];duality=[];replays=[];stateerrors=[]
 for l in [1,2]:
  f=data['fits'][l];x=data['captures']['train',l].double().cuda();u=x/(x.square().mean(1,keepdim=True)+eps).sqrt();val=data['captures']['validation',l].double().cuda();vr=(val.square().mean(1,keepdim=True)+eps).sqrt();vu=val/vr;tmean=f['token_shape_mean'].cuda();lookup={int(t):i for i,t in enumerate(f['token_ids'])};tridx=torch.tensor([lookup[int(t)] for t in data['train_token_ids']],device='cuda');tu=torch.stack([tmean[lookup[int(t)]] if int(t) in lookup else f['shape_mean'].cuda() for t in data['validation_token_ids']]);res=u-tmean[tridx];C=res.T@res/len(res);_,P=torch.linalg.eigh(C);P=P[:,-64:].flip(1)
  mlp=model.blocks[l].mlp;assert not mlp.config.gated;L,R,D=[getattr(mlp,n).weight.double() for n in ['Left','Right','Down']];gen=torch.Generator(device='cuda').manual_seed(833+l);probes=torch.linalg.qr(torch.randn(1152,8,dtype=torch.float64,device='cuda',generator=gen)).Q.T;W=torch.zeros(1152,1152,dtype=torch.float64,device='cuda')
  for v in probes:
   lv=L@v;rv=R@v;T=D@(rv[:,None]*L+lv[:,None]*R);W+=T.T@T/8
   s=u[:3];direct=F.linear(F.linear(s,L)*rv+lv*F.linear(s,R),D);metricerr.append(float((s@T.T-direct).norm()/direct.norm()))
  ridge=.001*W.trace()/1152;W+=ridge*torch.eye(1152,device='cuda');ev,Q=torch.linalg.eigh(W);assert ev.min()>0;root=(Q*ev.sqrt())@Q.T;invroot=(Q*ev.rsqrt())@Q.T;_,V=torch.linalg.eigh(root@C@root);B=V[:,-64:].flip(1);E=root@B;Fdec=invroot@B;duality.append(float((E.T@Fdec-torch.eye(64,device='cuda')).abs().max()));pred={}
  for rank in [4,16,64]:
   for name,en,de in [('residual_pca',P[:,:rank],P[:,:rank]),('response_weighted',E[:,:rank],Fdec[:,:rank])]:
    shape=tu+(vu-tu)@en@de.T;pred[f'{name}{rank}']=shape/shape.square().mean(1,keepdim=True).sqrt().clamp_min(1e-30)*(vr.square()-eps).sqrt();stateerrors.append({'layer':l,'model':f'{name}{rank}','shape_relative_RMSE':float((shape-vu).norm()/vu.norm())})
  def response(base,v):
   r=(base.square().mean(1,keepdim=True)+eps).sqrt();s=base/r;q=v/r-base*(base*v).mean(1,keepdim=True)/r.pow(3);return F.linear(F.linear(s,L)*F.linear(q,R)+F.linear(q,L)*F.linear(s,R),D)
  for di,v in enumerate(f['directions'].cuda()):
   truth=response(val,v);den=truth.square().sum(1)
   for name,p in pred.items():
    y=response(p,v);err=(y-truth).square().sum(1);co=(y*truth).sum(1)/(y.norm(dim=1)*truth.norm(dim=1)).clamp_min(1e-30)
    for i in range(512):records.append({'layer':l,'document':data['validation_documents'][i],'probe':di,'model':name,'squared_error':float(err[i]),'truth_squared_norm':float(den[i]),'cosine':float(co[i])})
   orig=[r for r in previous['records'] if r['layer']==l and r['model']=='pca64_radius' and r['probe']==di];replays.append(max(abs(float(den[i])-orig[i]['truth_squared_norm'])/max(orig[i]['truth_squared_norm'],1e-30) for i in range(512)))
  fits[l]={'residual_pca':P.cpu(),'weighted_encoder':E.cpu(),'weighted_decoder':Fdec.cpu(),'fit_probes':probes.cpu(),'ridge':float(ridge),'metric_eigenvalues':ev.cpu()};print('DONE',l,flush=True)
 summary=[]
 for l in [1,2]:
  for name in cfg['models']:
   rr=[r for r in records if r['layer']==l and r['model']==name];summary.append({'layer':l,'model':name,'relative_RMSE':float(np.sqrt(sum(r['squared_error'] for r in rr)/sum(r['truth_squared_norm'] for r in rr))),'mean_cosine':float(np.mean([r['cosine'] for r in rr]))})
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'metric_linear_map_relative_error_max':max(metricerr),'weighted_encoder_decoder_duality_error_max':max(duality),'truth_norm_replay_relative_max':max(replays),'finite':all(np.isfinite(r['relative_RMSE']) for r in summary),'whole_model_forwards':0,'test_evaluations':0};assert audits['hashes_unchanged'] and audits['finite'] and max(metricerr)<1e-10 and max(duality)<1e-8 and max(replays)<1e-10;torch.save({'fits':fits},ART/f'{S}_fits.pt');(ART/f'{S}.json').write_text(json.dumps({'config':cfg,'records':records,'stateerrors':stateerrors,'summary':summary,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(json.dumps({'summary':summary,'audits':audits},indent=2),flush=True)
if __name__=='__main__':main()

"""H381 frozen shared versus recipient-specific coefficient metrics."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,B
STEM=ART/'bilin18_l17_shared_metric_coefficients'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,digest in cfg['source_sha256'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==digest
 assert hashlib.sha256((ROOT/cfg['corpus']).read_bytes()).hexdigest()==cfg['corpus_sha256']
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float()
 pre,_,_=collect_sources(model,np.load(ROOT/cfg['corpus'],mmap_mode='r'),cfg)
 L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]]
 x=F.rms_norm(pre,(1152,));state=pre+B(x,x,L,R,D)+model.blocks[17].mlp.Down_bias;baseline=F.rms_norm(state,(1152,))
 W=model.lm_head.weight.double();assert model.lm_head.bias is None
 frozen=np.load(ART/'bilin18_l17_probability_axes_finite_frozen_basis.npz');K=torch.tensor(frozen['metric'],device='cuda');bases=[torch.tensor(frozen['eigenvectors'][:,:64],device='cuda'),torch.tensor(np.load(ART/'bilin18_l17_random_span_specificity_seed20270602_basis.npz')['basis'],device='cuda')];S=len(pre);deltas=[]
 for h in cfg['heads']:
  responses=np.load(ART/f'bilin18_l17_shared_geometry_precision_h{h}_responses.npz')['responses'];deltas.extend([(F.rms_norm(state+torch.tensor(v,device='cuda'),(1152,))-baseline).double() for v in responses])
 deltas=torch.stack(deltas);grid=cfg['step_grid'];allbest=[];audits=[]
 for bi,V in enumerate(bases):
  mapped=V.T@W.T;Gshared=V.T@K@V;maps=[torch.linalg.solve(Gshared[:rank,:rank],(K@V[:,:rank]).T).T for rank in cfg['ranks']];eigenaudit=max(float((maps[i]-V[:,:rank]).abs().max()) for i,rank in enumerate(cfg['ranks'])) if bi==0 else None
  if bi==0:assert eigenaudit<1e-10
  kl=np.empty((39,2,2,len(grid),S));stationarity=0.;nonoptimal=0.;minrank=[16,64]
  for off in range(0,S,8):
   sl=slice(off,off+8);d=deltas[:,sl].transpose(0,1);raw=baseline[sl].double()@W.T;dlog=d@W.T;bp=(30*torch.tanh(raw/30)).softmax(-1);J=mapped[None]-(mapped[None]*bp[:,None]).sum(-1,keepdim=True);G=(J*bp[:,None])@J.transpose(1,2);tj=dlog-(dlog*bp[:,None]).sum(-1,keepdim=True);rhs=(tj*bp[:,None])@J.transpose(1,2);targetlp=(30*torch.tanh((raw[:,None]+dlog)/30)).log_softmax(-1);tp=targetlp.exp()
   for ri,rank in enumerate(cfg['ranks']):
    sharedcoef=d@maps[ri];g=G[:,:rank,:rank];val,vec=torch.linalg.eigh((g+g.transpose(1,2))/2);keep=val>val[:,-1:]*cfg['pinv_rtol'];minrank[ri]=min(minrank[ri],int(keep.sum(-1).min()));iv=torch.where(keep,1/val.clamp_min(1e-300),0);b=rhs[:,:,:rank];owncoef=((b@vec)*iv[:,None])@vec.transpose(1,2);stationarity=max(stationarity,float(((owncoef@g-b).norm(dim=-1)/b.norm(dim=-1).clamp_min(1e-30)).max()));errs=[]
    for mi,coef in enumerate([sharedcoef,owncoef]):
     change=coef@mapped[:rank];err=dlog-change;err-= (err*bp[:,None]).sum(-1,keepdim=True);errs.append((bp[:,None]*err.square()).sum(-1))
     for gi,t in enumerate(grid):
      lp=(30*torch.tanh((raw[:,None]+t*change)/30)).log_softmax(-1);kl[:,mi,ri,gi,sl]=(tp*(targetlp-lp)).sum(-1).T.cpu().numpy()
    den=(bp[:,None]*tj.square()).sum(-1);nonoptimal=max(nonoptimal,float(((errs[1]-errs[0])/den).max()))
   if off%512==0:print('basis',bi,'sites',off+8,'of',S,flush=True)
  kl=kl.reshape(3,13,2,2,len(grid),S);best=kl.min(4);assert np.isfinite(kl).all() and kl.min()>-1e-10 and nonoptimal<1e-7 and stationarity<1e-7
  np.savez_compressed(STEM.with_name(STEM.name+f'_basis{bi}_rows.npz'),grid_kl=kl,best_kl=best,chosen_index=kl.argmin(4));allbest.append(best);audits.append({'basis':bi,'probability_eigenmap_replay_max':eigenaudit,'normal_equation_max_relative':stationarity,'max_own_uncapped_suboptimality':nonoptimal,'minimum_retained_ranks':minrank})
 analyze(cfg,np.stack(allbest),audits)
def analyze(cfg,best,audits):
 n=cfg['rows'];groups=[slice(0,1),slice(1,7),slice(7,13)];capped=np.stack([np.load(ART/'bilin18_l17_oracle_finite_safeguard_rows.npz')['best_kl'],np.load(ART/'bilin18_l17_random_span_specificity_seed20270602_rows.npz')['best_kl']]);errors=np.concatenate([best,capped[:,:,:,None]],axis=3).reshape(2,3,13,3,2,n,4);base=np.load(ART/'bilin18_l17_probability_axes_oracle_rows.npz')['kl'][:,:,0].reshape(3,13,n,4)
 rec=np.stack([1-errors[:,:,g].mean((2,5,6))/base[:,g].mean((1,2,3))[None,:,None,None] for g in groups],2);gain=rec[:,:,:,1]-rec[:,:,:,0];capgap=rec[:,:,:,2]-rec[:,:,:,1];rng=np.random.default_rng(cfg['bootstrap_seed']);br=[];bg=[];bc=[]
 for _ in range(cfg['bootstrap_samples']):
  ix=rng.integers(0,n,n);rr=[]
  for g in groups:
   e=errors[:,:,g];b=base[:,g];draw=rng.integers(0,e.shape[2],e.shape[2]);em=e[:,:,draw][:,:,:,:,:,ix].mean((2,5,6));bm=b[:,draw][:,:,ix].mean((1,2,3));rr.append(1-em/bm[None,:,None,None])
  rr=np.stack(rr,2);br.append(rr[:,:,:,0]);bg.append(rr[:,:,:,1]-rr[:,:,:,0]);bc.append(rr[:,:,:,2]-rr[:,:,:,1])
 def band(b,t):
  b=np.stack(b);rad=float(np.quantile(np.max(np.abs(b-t),axis=tuple(range(1,b.ndim))),.95));return np.stack([t-rad,t+rad],-1)
 rci=band(br,rec[:,:,:,0]);gci=band(bg,gain);cci=band(bc,capgap);gates={'probability_adaptation10':bool((gci[:,:,:,:,0]>.1).all()),'cap_equivalent5':bool((cci[:,:,:,:,0]>=-.05).all() and (cci[:,:,:,:,1]<=.05).all()),'shared64_half':bool((rci[:,:,:,1,0]>.5).all())};j={'config':cfg,'basis_order':['probability','random'],'metric_order':['shared','own_uncapped','own_capped'],'families':['actual','moment','donor'],'recovery':rec.tolist(),'shared_joint36_ci95':rci.tolist(),'own_uncapped_minus_shared':gain.tolist(),'adaptation_joint36_ci95':gci.tolist(),'own_capped_minus_uncapped':capgap.tolist(),'cap_joint36_ci95':cci.tolist(),'audits':audits,'gates':gates};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');np.savez_compressed(STEM.with_name(STEM.name+'_bootstrap.npz'),shared=br,adaptation=bg,cap=bc);print('FINISHED',gates,'recovery',rec,flush=True)
if __name__=='__main__':main()

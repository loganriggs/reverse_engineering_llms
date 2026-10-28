"""H376 full-panel finite step-size safeguard for local oracle."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,B
STEM=ART/'bilin18_l17_oracle_finite_safeguard'
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
 V=torch.tensor(np.load(ART/'bilin18_l17_probability_axes_finite_frozen_basis.npz')['eigenvectors'][:,:64],device='cuda');mapped=V.T@W.T;S=len(pre);deltas=[]
 for h in cfg['heads']:
  responses=np.load(ART/f'bilin18_l17_shared_geometry_precision_h{h}_responses.npz')['responses'];deltas.extend([(F.rms_norm(state+torch.tensor(v,device='cuda'),(1152,))-baseline).double() for v in responses])
 deltas=torch.stack(deltas);old=np.load(ART/'bilin18_l17_probability_axes_oracle_rows.npz');coeff=torch.tensor(old['oracle_coefficients'],device='cuda');grid=cfg['step_grid'];kl=np.empty((39,2,len(grid),S));replay=0.
 for off in range(0,S,8):
  sl=slice(off,off+8);d=deltas[:,sl].transpose(0,1);raw=baseline[sl].double()@W.T;targetlp=(30*torch.tanh((raw[:,None]+d@W.T)/30)).log_softmax(-1);tp=targetlp.exp()
  for ri,rank in enumerate(cfg['ranks']):
   change=coeff[:,ri,sl,:rank].transpose(0,1)@mapped[:rank]
   for gi,t in enumerate(grid):
    patchlp=(30*torch.tanh((raw[:,None]+t*change)/30)).log_softmax(-1);kl[:,ri,gi,sl]=(tp*(targetlp-patchlp)).sum(-1).T.cpu().numpy()
  if off%512==0:print('sites',off+8,'of',S,flush=True)
 kl=kl.reshape(3,13,2,len(grid),S);choice=kl.argmin(3);best=kl.min(3);original=old['kl'][:,:,[3,4]];baselinekl=old['kl'][:,:,0];replay=max(float(np.max(np.abs(kl[:,:,:,grid.index(1.)]-original))/np.max(original)),float(np.max(np.abs(kl[:,:,:,0]-baselinekl[:,:,None]))/np.max(baselinekl)));assert replay<1e-8 and np.isfinite(kl).all() and kl.min()>-1e-10 and np.max(best-baselinekl[:,:,None])<1e-10 and np.max(best-original)<1e-10
 np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),grid_kl=kl,best_kl=best,chosen_index=choice)
 analyze(cfg,kl,best,choice,old,{'h374_endpoint_replay_relative':replay,'max_best_minus_baseline':float(np.max(best-baselinekl[:,:,None])),'max_best_minus_original':float(np.max(best-original))})
def analyze(cfg,kl,best,choice,old,audit):
 groups=[slice(0,1),slice(1,7),slice(7,13)];n=cfg['rows'];kv=best.reshape(3,13,2,n,4);bv=old['kl'][:,:,0].reshape(3,13,n,4);ov=old['kl'][:,:,[3,4]].reshape(kv.shape);lv=old['local_error'][:,:,[3,4]].reshape(kv.shape);lb=old['local_error'][:,:,0].reshape(bv.shape)
 def mean(v):return np.stack([v[:,g].mean((1,)+tuple(range(v.ndim-2,v.ndim))) for g in groups],1)
 bm=mean(bv);rec=1-mean(kv)/bm[:,:,None];original=1-mean(ov)/bm[:,:,None];local=1-mean(lv)/mean(lb)[:,:,None];rng=np.random.default_rng(cfg['bootstrap_seed']);bs=[]
 for _ in range(cfg['bootstrap_samples']):
  ix=rng.integers(0,n,n);kk=[];bb=[]
  for g in groups:
   k=kv[:,g];b=bv[:,g];rr=rng.integers(0,k.shape[1],k.shape[1]);kk.append(k[:,rr][:,:,:,ix].mean((1,3,4)));bb.append(b[:,rr][:,:,ix].mean((1,2,3)))
  k=np.stack(kk,1);b=np.stack(bb,1);bs.append(1-k/b[:,:,None])
 bs=np.stack(bs);rad=float(np.quantile(np.max(np.abs(bs-rec),axis=(1,2,3)),.95));ci=np.stack([rec-rad,rec+rad],-1);diff=float(np.max(np.abs(rec-local)));gates={'guard16_sufficient':bool((ci[:,:,0,0]>=.8).all()),'guard64_sufficient':bool((ci[:,:,1,0]>=.8).all()),'local_finite_agreement':diff<.05};hist=np.stack([np.stack([(choice[:,g]==i).mean((1,3)) for i in range(len(cfg['step_grid']))],-1) for g in groups],1)
 worse=ov>bv[:,:,None];worsefrac=mean(worse.astype(float));j={'config':cfg,'families':['actual','moment','donor'],'guarded_recovery':rec.tolist(),'unguarded_recovery':original.tolist(),'unguarded_local_recovery':local.tolist(),'guarded_joint18_ci95':ci.tolist(),'max_guarded_minus_local_absolute':diff,'chosen_step_fractions':hist.tolist(),'unguarded_worse_than_baseline_fraction':worsefrac.tolist(),'audit':audit,'gates':gates};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');np.savez_compressed(STEM.with_name(STEM.name+'_bootstrap.npz'),recovery=bs);print('FINISHED',gates,'guarded',rec,'maxlocaldiff',diff,flush=True)
if __name__=='__main__':main()

"""H407 baseline-preserving factor clamp."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,B
from run_bilin18_l17_native_sensitivity_factors import rotate
STEM=ART/'bilin18_l17_baseline_preserving_factor_clamp'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,d in cfg['source_sha256'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==d
 assert hashlib.sha256((ROOT/cfg['corpus']).read_bytes()).hexdigest()==cfg['corpus_sha256']
 docs=np.load(ROOT/cfg['corpus']);labels=torch.tensor(docs[:,np.array(cfg['positions'])+1].reshape(-1),device='cuda',dtype=torch.long)
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();pre,heads,_=collect_sources(model,docs,cfg);S=len(pre);L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];W=model.lm_head.weight.double();x=F.rms_norm(pre,(1152,));phi0=(x@L.T)*(x@R.T);m0=phi0@D.T;state=pre+m0+model.blocks[17].mlp.Down_bias;orders=np.load(ART/'bilin18_l17_native_sensitivity_factors_frozen_orders.npz')['orders'];ids=[torch.tensor(o[:64],device='cuda') for o in orders];removed0=torch.stack([phi0[:,ix]@D[:,ix].T for ix in ids]);deleted0=F.rms_norm(state[None]-removed0,(1152,));base=F.rms_norm(state,(1152,))
 def lp(y):return (30*torch.tanh((y.double()@W.T)/30)).log_softmax(-1)
 baseline_error=0.
 for ix in ids:
  unchanged=phi0.clone();unchanged[:,ix]=phi0[:,ix];baseline_error=max(baseline_error,float((F.rms_norm(pre+unchanged@D.T+model.blocks[17].mlp.Down_bias,(1152,))-base).abs().max()))
 kl=np.empty((8,7,2,3,S));effect=np.empty((7,2,S));rawerr=np.empty((8,7,2,S));rawenergy=np.empty((7,2,S));audit=0.;geometry=[];gram=heads.double()@heads.double().transpose(-1,-2);hn=heads.double().norm(dim=-1);pu=pre.double()/pre.double().norm(dim=-1,keepdim=True);dots=(heads.double()*pu[:,None]).sum(-1)
 for ci,seed in enumerate([None]+cfg['rotation_seeds']):
  H=rotate(pre,heads,seed)
  if seed is not None:
   ge=float(((H.double()@H.double().transpose(-1,-2)-gram).abs()/(hn[:,:,None]*hn[:,None])).max());de=float((((H.double()*pu[:,None]).sum(-1)-dots).abs()/hn).max());assert max(ge,de)<1e-6;geometry.append([ge,de])
  for ai,amp in enumerate(cfg['amplitudes']):
   xx=F.rms_norm(pre-amp*H[:,6],(1152,));phi1=(xx@L.T)*(xx@R.T);r=phi1@D.T-m0;delta=phi1-phi0;kept=torch.stack([delta[:,ix]@D[:,ix].T for ix in ids]);target=F.rms_norm(state+r,(1152,));prediction=F.rms_norm(state[None]+kept,(1152,));deleted1=F.rms_norm(state[None]-removed0+r[None]-kept,(1152,));rawenergy[ci,ai]=r.double().square().sum(-1).cpu().numpy();rawerr[:,ci,ai]=(kept.double()-r.double()[None]).square().sum(-1).cpu().numpy()
   clamped=F.rms_norm(state[None]+r[None]-kept,(1152,));patched=phi1[:4].clone();patched[:,ids[0]]=phi0[:4,ids[0]];direct=F.rms_norm(pre[:4]+patched@D.T+model.blocks[17].mlp.Down_bias,(1152,));audit=max(audit,float((direct-clamped[0,:4]).abs().max()))
   for off in range(0,S,16):
    sl=slice(off,off+16);p0=lp(base[sl]);pt=lp(target[sl]);pc=lp(clamped[:,sl]);tp=pt.exp();effect[ci,ai,sl]=(tp*(pt-p0)).sum(-1).cpu().numpy();kl[:,ci,ai,0,sl]=(pc.exp()*(pc-p0[None])).sum(-1).cpu().numpy();kl[:,ci,ai,1,sl]=(tp[None]*(pt[None]-pc)).sum(-1).cpu().numpy();lab=labels[sl];kl[:,ci,ai,2,sl]=(p0.gather(-1,lab[:,None]).squeeze(-1)[None]-pc.gather(-1,lab[None,:,None].expand(8,-1,1)).squeeze(-1)).cpu().numpy()
   print('condition',ci,'amplitude',amp,'clamp forward audit',audit,flush=True)
  np.savez_compressed(STEM.with_name(STEM.name+'_partial_rows.npz'),kl=kl[:,:ci+1],native_effect=effect[:ci+1],raw_error=rawerr[:,:ci+1],raw_energy=rawenergy[:ci+1])
 assert np.isfinite(kl).all() and kl[:,:,:,0:2].min()>-1e-10 and effect.min()>0
 np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),kl=kl,native_effect=effect,raw_error=rawerr,raw_energy=rawenergy);analyze(cfg,kl,effect,{'max_clamp_forward_state_absolute_error':audit,'baseline_state_absolute_error':baseline_error,'geometry_errors':geometry})
def analyze(cfg,kl,effect,audit):
 n=cfg['rows'];old=np.load(ART/'bilin18_l17_finite_sensitivity_factors_rows.npz');audit['native_effect_replay_max_absolute']=float(abs(effect-old['native_effect']).max());k=kl.reshape(8,7,2,3,n,4).mean(-1);e=effect.reshape(7,2,n,4).mean(-1);deleted=old['kl'].reshape(8,7,2,3,n,4).mean(-1)
 def summary(ix,randoms):
  den=e[0,0,ix].mean();actual=k[0,0,0,0,ix].mean()/den;random=k[randoms,0,0,0][:,ix].mean()/den;deletion=deleted[0,0,0,1,ix].mean()/den;return np.array([actual,random-actual,actual-deletion])
 point=summary(np.arange(n),np.arange(2,8));rng=np.random.default_rng(cfg['bootstrap_seed']);boot=np.stack([summary(rng.integers(0,n,n),2+rng.integers(0,6,6)) for _ in range(cfg['bootstrap_samples'])]);rad=np.quantile(abs(boot-point).max(1),.95);ci=np.stack([point-rad,point+rad],-1);gates={'baseline_identity':audit['baseline_state_absolute_error']==0,'native_replay':audit['native_effect_replay_max_absolute']<1e-10,'remaining_unit20':bool(ci[0,1]<.2),'over_random10':bool(ci[1,0]>.1),'deletion_equivalence05':bool(ci[2,0]>-.05 and ci[2,1]<.05)};km=np.stack([k[:,0].mean(-1),k[:,1:].mean((1,4))],1);em=np.stack([e[0].mean(-1),e[1:].mean((0,2))]);j={'config':cfg,'families':['actual','rotated'],'methods':['contrast64','energy64']+[f'random64_{i}' for i in range(6)],'metrics':['clamped_remaining_KL','native_vs_clamped_KL','clamped_NLL_minus_intact_NLL'],'mean_metrics':km.tolist(),'mean_native_effect':em.tolist(),'primary_names':['actual_unit_clamped_remaining_fraction','random_minus_contrast_remaining_fraction','clamp_minus_deletion_remaining_fraction'],'primary':point.tolist(),'primary_joint3_ci95':ci.tolist(),'audit':audit,'gates':gates};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');np.savez_compressed(STEM.with_name(STEM.name+'_bootstrap.npz'),primary=boot);print('FINISHED',gates,point,flush=True)
if __name__=='__main__':main()

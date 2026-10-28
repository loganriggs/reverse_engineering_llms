"""H408 native-output Fisher factor ranking; frozen prior to held-out evaluation."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources
from run_bilin18_l17_native_sensitivity_factors import drms
STEM=ART/'bilin18_l17_fisher_ranked_factors'
def native_drms(p,h):
 s=(p.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt()
 return h/s-p*(p*h).mean(-1,keepdim=True)/s.pow(3)
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for k,v in cfg['corpus_sha256'].items():assert hashlib.sha256((ROOT/cfg[k]).read_bytes()).hexdigest()==v
 oldpath=ART/'bilin18_l17_native_sensitivity_factors_frozen_orders.npz';assert hashlib.sha256(oldpath.read_bytes()).hexdigest()==cfg['orders_sha256']
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];Dd=D.double();W=model.lm_head.weight.double()
 def lp(y):return (30*torch.tanh((y.double()@W.T)/30)).log_softmax(-1)
 audit={};orders=None
 for stage in ['discovery','evaluation']:
  cc=dict(cfg,corpus=cfg[stage+'_corpus']);pre,heads,_=collect_sources(model,np.load(ROOT/cc['corpus']),cc);S=len(pre);x=F.rms_norm(pre,(1152,));phi0=(x@L.T)*(x@R.T);m0=phi0@D.T;state=pre+m0+model.blocks[17].mlp.Down_bias;base=F.rms_norm(state,(1152,))
  if stage=='discovery':
   u=drms(pre,-heads[:,6]);phi=((u@L.T)*(x@R.T)+(x@L.T)*(u@R.T)).double();score=torch.zeros(4608,device='cuda',dtype=torch.float64);closure=0.;fisher=[]
   for off in range(0,S,16):
    sl=slice(off,off+16);s=state[sl].double();g=phi[sl]@Dd.T;y=native_drms(s,g);raw=base[sl].double()@W.T;tanh=torch.tanh(raw/30);cap=1-tanh.square();p=(30*tanh).softmax(-1);t=(y@W.T)*cap;center=t-(p*t).sum(-1,keepdim=True);v=(p*center*cap)@W;pull=native_drms(s,v);attr=phi[sl]*(pull@Dd);metric=(p*center.square()).sum(-1);closure=max(closure,float(((attr.sum(-1)-metric).abs()/metric).max()));score+=attr.sum(0);fisher.append(metric.cpu().numpy())
   score=(score/S).cpu().numpy();orders=np.concatenate([np.argsort(-score,kind='stable')[None],np.load(oldpath)['orders']],0);np.savez_compressed(STEM.with_name(STEM.name+'_frozen_orders.npz'),orders=orders,fisher_signed_scores=score,discovery_fisher=np.concatenate(fisher));audit['fisher_attribution_relative_closure']=closure;assert closure<1e-8;print('FROZEN',hashlib.sha256(STEM.with_name(STEM.name+'_frozen_orders.npz').read_bytes()).hexdigest(),'closure',closure,flush=True);continue
  ids=[torch.tensor(order[:rank],device='cuda') for order in orders for rank in cfg['ranks']];N=len(ids);metrics=np.empty((N,2,2,S));effect=np.empty((2,S));rawerr=np.empty((N,2,S));rawenergy=np.empty((2,S));direct_error=0.
  for ai,amp in enumerate(cfg['amplitudes']):
   xx=F.rms_norm(pre-amp*heads[:,6],(1152,));phi1=(xx@L.T)*(xx@R.T);r=phi1@D.T-m0;delta=phi1-phi0;kept=torch.stack([delta[:,ix]@D[:,ix].T for ix in ids]);target=F.rms_norm(state+r,(1152,));pred=F.rms_norm(state[None]+kept,(1152,));clamp=F.rms_norm(state[None]+r[None]-kept,(1152,));rawenergy[ai]=r.double().square().sum(-1).cpu().numpy();rawerr[:,ai]=(kept.double()-r.double()[None]).square().sum(-1).cpu().numpy()
   for oi in [0,1,2]:
    patched=phi1[:4].clone();patched[:,ids[oi]]=phi0[:4,ids[oi]];direct=F.rms_norm(pre[:4]+patched@D.T+model.blocks[17].mlp.Down_bias,(1152,));direct_error=max(direct_error,float((direct-clamp[oi,:4]).abs().max()))
   for off in range(0,S,16):
    sl=slice(off,off+16);pt=lp(target[sl]);p0=lp(base[sl]);effect[ai,sl]=(pt.exp()*(pt-p0)).sum(-1).cpu().numpy()
    for oi in range(N):
     pp=lp(pred[oi,sl]);pc=lp(clamp[oi,sl]);metrics[oi,ai,0,sl]=(pt.exp()*(pt-pp)).sum(-1).cpu().numpy();metrics[oi,ai,1,sl]=(pc.exp()*(pc-p0)).sum(-1).cpu().numpy()
   print('evaluation amplitude',amp,'complete',flush=True)
  metrics=metrics.reshape(9,3,2,2,S);rawerr=rawerr.reshape(9,3,2,S);assert np.isfinite(metrics).all() and metrics.min()>-1e-10
  np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),metrics=metrics,native_effect=effect,raw_error=rawerr,raw_energy=rawenergy);audit['direct_clamp_state_absolute_error']=direct_error
 analyze(cfg,metrics,effect,audit)
def analyze(cfg,m,e,audit):
 old=np.load(ART/'bilin18_l17_finite_sensitivity_factors_rows.npz');cl=np.load(ART/'bilin18_l17_baseline_preserving_factor_clamp_rows.npz');audit['native_effect_replay_max_absolute']=float(abs(e-old['native_effect'][0]).max());audit['contrast_energy_random64_prediction_replay']=float(abs(m[1:,1,:,0]-old['kl'][:,0,:,0]).max());audit['contrast_energy_random64_clamp_replay']=float(abs(m[1:,1,:,1]-cl['kl'][:,0,:,0]).max());n=cfg['rows'];d=m.reshape(9,3,2,2,n,4).mean(-1);ed=e.reshape(2,n,4).mean(-1)
 def summary(ix):
  den=ed[0,ix].mean();a=1-d[0,1,0,0,ix].mean()/den;b=d[0,1,0,1,ix].mean()/den;c=(d[1,1,0,0,ix].mean()-d[0,1,0,0,ix].mean())/den;return np.array([a,b,c])
 point=summary(np.arange(n));rng=np.random.default_rng(cfg['bootstrap_seed']);boot=np.stack([summary(rng.integers(0,n,n)) for _ in range(cfg['bootstrap_samples'])]);rad=np.quantile(abs(boot-point).max(1),.95);ci=np.stack([point-rad,point+rad],-1);gates={'fisher_closure':audit['fisher_attribution_relative_closure']<1e-8,'native_replay':audit['native_effect_replay_max_absolute']<1e-10,'native_recovery90':bool(ci[0,0]>.9),'clamp_remaining20':bool(ci[1,1]<.2),'improvement05':bool(ci[2,0]>.05)}
 j={'config':cfg,'methods':['fisher','contrast','individual_energy']+[f'random{i}' for i in range(6)],'metrics':['retained_prediction_KL','clamped_remaining_KL'],'mean_metrics':m.mean(-1).tolist(),'mean_native_effect':e.mean(-1).tolist(),'primary_names':['fisher64_unit_recovery','fisher64_unit_clamp_remaining','fisher64_minus_contrast64_unit_recovery'],'primary':point.tolist(),'primary_joint3_ci95':ci.tolist(),'gates':gates,'audit':audit};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');np.savez_compressed(STEM.with_name(STEM.name+'_bootstrap.npz'),primary=boot);print('FINISHED',gates,point,flush=True)
if __name__=='__main__':main()

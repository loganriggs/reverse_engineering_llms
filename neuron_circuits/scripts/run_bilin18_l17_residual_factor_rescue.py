"""H410 one-step residual-correlation factor rescue, no refit."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources
from run_bilin18_l17_native_sensitivity_factors import drms
from run_bilin18_l17_fisher_ranked_factors import native_drms
STEM=ART/'bilin18_l17_residual_factor_rescue'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());op=ART/'bilin18_l17_fisher_ranked_factors_frozen_orders.npz';assert hashlib.sha256(op.read_bytes()).hexdigest()==cfg['orders_sha256']
 for k,v in cfg['corpus_sha256'].items():assert hashlib.sha256((ROOT/cfg[k]).read_bytes()).hexdigest()==v
 orders=np.load(op)['orders'];core=orders[0,:64];coreids=torch.tensor(core,device='cuda');model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];Dd=D.double();W=model.lm_head.weight.double();trows=[];frows=[];audit={}
 def lp(y):return (30*torch.tanh((y.double()@W.T)/30)).log_softmax(-1)
 for stage in ['discovery','evaluation']:
  cc=dict(cfg,corpus=cfg[stage+'_corpus']);pre,heads,_=collect_sources(model,np.load(ROOT/cc['corpus']),cc);S=len(pre);x=F.rms_norm(pre,(1152,));phi0=(x@L.T)*(x@R.T);m0=phi0@D.T;state=pre+m0+model.blocks[17].mlp.Down_bias;base=F.rms_norm(state,(1152,));u=drms(pre,-heads[:,6]);phi=((u@L.T)*(x@R.T)+(x@L.T)*(u@R.T)).double();full=phi@Dd.T
  if stage=='discovery':
   residual=full-phi[:,coreids]@Dd[:,coreids].T;score=torch.zeros(4608,device='cuda',dtype=torch.float64);omitted_energy=0.
   for off in range(0,S,16):
    sl=slice(off,off+16);s=state[sl].double();th=torch.tanh((base[sl].double()@W.T)/30);cap=1-th.square();p=(30*th).softmax(-1);t=(native_drms(s,residual[sl])@W.T)*cap;t-=(p*t).sum(-1,keepdim=True);pull=native_drms(s,(p*t*cap)@W);score+=(phi[sl]*(pull@Dd)).sum(0);omitted_energy+=float((p*t.square()).sum())
   score=(score/S).cpu().numpy();allowed=np.setdiff1d(np.arange(4608),core);ranking=allowed[np.argsort(-score[allowed],kind='stable')];sets=[core,np.concatenate([core,ranking[:64]]),orders[0,:128],orders[1,:128],orders[2,:128]]+[np.concatenate([core,np.random.default_rng(seed).permutation(allowed)[:64]]) for seed in cfg['random_seeds']];audit['residual_score_energy_relative_closure']=abs(score[allowed].sum()-omitted_energy/S)/(omitted_energy/S);assert audit['residual_score_energy_relative_closure']<1e-8;np.savez_compressed(STEM.with_name(STEM.name+'_frozen_sets.npz'),core=core,sets128=np.stack(sets[1:]),residual_signed_score=score,omitted_ranking=ranking);print('FROZEN SETS',hashlib.sha256(STEM.with_name(STEM.name+'_frozen_sets.npz').read_bytes()).hexdigest(),flush=True)
  ids=[torch.tensor(v,device='cuda') for v in sets];kept=torch.stack([phi[:,ix]@Dd[:,ix].T for ix in ids]);tm=np.empty((11,4,S));fe=np.empty(S);closure=0.
  for off in range(0,S,16):
   sl=slice(off,off+16);s=state[sl].double();th=torch.tanh((base[sl].double()@W.T)/30);cap=1-th.square();p=(30*th).softmax(-1);t=(native_drms(s,full[sl])@W.T)*cap;t-=(p*t).sum(-1,keepdim=True);f=(p*t.square()).sum(-1);fe[sl]=f.cpu().numpy()
   for oi in range(11):
    k=(native_drms(s,kept[oi,sl])@W.T)*cap;k-=(p*k).sum(-1,keepdim=True);o=t-k;ke=(p*k.square()).sum(-1);oe=(p*o.square()).sum(-1);cross=2*(p*k*o).sum(-1);at=(p*k*t).sum(-1);tm[oi,:,sl]=torch.stack([ke,oe,cross,at]).cpu().numpy();closure=max(closure,float(((ke+oe+cross-f).abs()/f).max()))
  trows.append(tm);frows.append(fe);audit[stage+'_energy_identity_relative_error']=closure;print(stage,'tangent complete',flush=True)
  if stage=='evaluation':
   metrics=np.empty((11,2,2,S));effect=np.empty((2,S));direct_error=0.
   for ai,amp in enumerate(cfg['amplitudes']):
    xx=F.rms_norm(pre-amp*heads[:,6],(1152,));phi1=(xx@L.T)*(xx@R.T);r=phi1@D.T-m0;delta=phi1-phi0;kept=torch.stack([delta[:,ix]@D[:,ix].T for ix in ids]);target=F.rms_norm(state+r,(1152,));pred=F.rms_norm(state[None]+kept,(1152,));clamp=F.rms_norm(state[None]+r[None]-kept,(1152,));patch=phi1[:4].clone();patch[:,ids[1]]=phi0[:4,ids[1]];direct=F.rms_norm(pre[:4]+patch@D.T+model.blocks[17].mlp.Down_bias,(1152,));direct_error=max(direct_error,float((direct-clamp[1,:4]).abs().max()))
    for off in range(0,S,16):
     sl=slice(off,off+16);p0=lp(base[sl]);pt=lp(target[sl]);effect[ai,sl]=(pt.exp()*(pt-p0)).sum(-1).cpu().numpy()
     for oi in range(11):
      pp=lp(pred[oi,sl]);pc=lp(clamp[oi,sl]);metrics[oi,ai,0,sl]=(pt.exp()*(pt-pp)).sum(-1).cpu().numpy();metrics[oi,ai,1,sl]=(pc.exp()*(pc-p0)).sum(-1).cpu().numpy()
    print('finite amplitude',amp,'complete',flush=True)
   audit['direct_clamp_state_absolute_error']=direct_error
 tm=np.stack(trows);fe=np.stack(frows);assert np.isfinite(tm).all() and np.isfinite(metrics).all();np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),tangent_metrics=tm,full_energy=fe,finite_metrics=metrics,native_effect=effect);analyze(cfg,tm,fe,metrics,effect,audit)
def analyze(cfg,tm,fe,m,e,audit):
 old=np.load(ART/'bilin18_l17_fisher_ranked_factors_rows.npz');prior=np.load(ART/'bilin18_l17_fisher_subset_diagnosis_rows.npz');audit['native_effect_replay_max_absolute']=float(abs(e-old['native_effect']).max());audit['core64_finite_replay_max_absolute']=float(abs(m[0]-old['metrics'][0,1]).max());audit['core64_tangent_replay_max_absolute']=float(abs(tm[:,0]-prior['metrics'][:,0,1]).max());n=cfg['rows'];d=m.reshape(11,2,2,n,4).mean(-1);ed=e.reshape(2,n,4).mean(-1)
 def summary(ix,randoms):
  den=ed[0,ix].mean();err=d[:,0,0][:,ix].mean(-1)/den;return np.array([1-err[1],err[2]-err[1],err[4]-err[1],err[randoms].mean()-err[1]])
 point=summary(np.arange(n),np.arange(5,11));rng=np.random.default_rng(cfg['bootstrap_seed']);boot=np.stack([summary(rng.integers(0,n,n),5+rng.integers(0,6,6)) for _ in range(cfg['bootstrap_samples'])]);rad=np.quantile(abs(boot-point).max(1),.95);ci=np.stack([point-rad,point+rad],-1);gates={'score_identity':bool(audit['residual_score_energy_relative_closure']<1e-8),'native_replay':audit['native_effect_replay_max_absolute']<1e-10,'recovery90':bool(ci[0,0]>.9),'over_fisher05':bool(ci[1,0]>.05),'over_energy02':bool(ci[2,0]>.02),'over_random05':bool(ci[3,0]>.05)};j={'config':cfg,'methods':['core64','residual128','fisher128','contrast128','individual_energy128']+[f'core_random128_{i}' for i in range(6)],'tangent_metrics':['retained_energy','omitted_energy','cross_energy','signed_attribution'],'mean_tangent_metrics':tm.mean(-1).tolist(),'mean_full_energy':fe.mean(-1).tolist(),'finite_metrics':['retained_prediction_KL','clamped_remaining_KL'],'mean_finite_metrics':m.mean(-1).tolist(),'mean_native_effect':e.mean(-1).tolist(),'primary_names':['residual128_unit_native_recovery','residual128_minus_fisher128_recovery','residual128_minus_energy128_recovery','residual128_minus_core_random128_recovery'],'primary':point.tolist(),'primary_joint4_ci95':ci.tolist(),'gates':gates,'audit':audit};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');np.savez_compressed(STEM.with_name(STEM.name+'_bootstrap.npz'),primary=boot);print('FINISHED',gates,point,flush=True)
if __name__=='__main__':main()

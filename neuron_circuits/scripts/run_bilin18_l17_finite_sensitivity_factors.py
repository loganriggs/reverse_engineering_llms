"""H406 native finite-factor confirmation, including intact output and NLL changes."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,B
from run_bilin18_l17_native_sensitivity_factors import rotate
STEM=ART/'bilin18_l17_finite_sensitivity_factors'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,d in cfg['source_sha256'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==d
 assert hashlib.sha256((ROOT/cfg['corpus']).read_bytes()).hexdigest()==cfg['corpus_sha256']
 docs=np.load(ROOT/cfg['corpus']);labels=torch.tensor(docs[:,np.array(cfg['positions'])+1].reshape(-1),device='cuda',dtype=torch.long)
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();pre,heads,_=collect_sources(model,docs,cfg);S=len(pre);L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];W=model.lm_head.weight.double();x=F.rms_norm(pre,(1152,));phi0=(x@L.T)*(x@R.T);m0=phi0@D.T;state=pre+m0+model.blocks[17].mlp.Down_bias;orders=np.load(ART/'bilin18_l17_native_sensitivity_factors_frozen_orders.npz')['orders'];ids=[torch.tensor(o[:64],device='cuda') for o in orders];removed0=torch.stack([phi0[:,ix]@D[:,ix].T for ix in ids]);deleted0=F.rms_norm(state[None]-removed0,(1152,));base=F.rms_norm(state,(1152,))
 def lp(y):return (30*torch.tanh((y.double()@W.T)/30)).log_softmax(-1)
 intact=np.empty((8,3,S));baseline_nll=np.empty(S)
 for off in range(0,S,16):
  sl=slice(off,off+16);p0=lp(base[sl]);pd=lp(deleted0[:,sl]);prob=p0.exp();intact[:,0,sl]=(prob[None]*(p0[None]-pd)).sum(-1).cpu().numpy();lab=labels[sl];n0=-p0.gather(-1,lab[:,None]).squeeze(-1);nd=-pd.gather(-1,lab[None,:,None].expand(8,-1,1)).squeeze(-1);intact[:,1,sl]=(nd-n0[None]).cpu().numpy();intact[:,2,sl]=nd.cpu().numpy();baseline_nll[sl]=n0.cpu().numpy()
 np.savez_compressed(STEM.with_name(STEM.name+'_intact_rows.npz'),metrics=intact,baseline_nll=baseline_nll)
 kl=np.empty((8,7,2,3,S));effect=np.empty((7,2,S));rawerr=np.empty((8,7,2,S));rawenergy=np.empty((7,2,S));audit=0.;geometry=[];gram=heads.double()@heads.double().transpose(-1,-2);hn=heads.double().norm(dim=-1);pu=pre.double()/pre.double().norm(dim=-1,keepdim=True);dots=(heads.double()*pu[:,None]).sum(-1)
 for ci,seed in enumerate([None]+cfg['rotation_seeds']):
  H=rotate(pre,heads,seed)
  if seed is not None:
   ge=float(((H.double()@H.double().transpose(-1,-2)-gram).abs()/(hn[:,:,None]*hn[:,None])).max());de=float((((H.double()*pu[:,None]).sum(-1)-dots).abs()/hn).max());assert max(ge,de)<1e-6;geometry.append([ge,de])
  for ai,amp in enumerate(cfg['amplitudes']):
   xx=F.rms_norm(pre-amp*H[:,6],(1152,));phi1=(xx@L.T)*(xx@R.T);r=phi1@D.T-m0;delta=phi1-phi0;kept=torch.stack([delta[:,ix]@D[:,ix].T for ix in ids]);target=F.rms_norm(state+r,(1152,));prediction=F.rms_norm(state[None]+kept,(1152,));deleted1=F.rms_norm(state[None]-removed0+r[None]-kept,(1152,));rawenergy[ci,ai]=r.double().square().sum(-1).cpu().numpy();rawerr[:,ci,ai]=(kept.double()-r.double()[None]).square().sum(-1).cpu().numpy()
   direct=F.rms_norm(pre[:4]+phi1[:4]@D.T+model.blocks[17].mlp.Down_bias-phi1[:4,ids[0]]@D[:,ids[0]].T,(1152,));audit=max(audit,float((direct-deleted1[0,:4]).abs().max()))
   for off in range(0,S,16):
    sl=slice(off,off+16);p0=lp(base[sl]);pt=lp(target[sl]);pp=lp(prediction[:,sl]);pd0=lp(deleted0[:,sl]);pd1=lp(deleted1[:,sl]);tp=pt.exp();effect[ci,ai,sl]=(tp*(pt-p0)).sum(-1).cpu().numpy();kl[:,ci,ai,0,sl]=(tp[None]*(pt[None]-pp)).sum(-1).cpu().numpy();kl[:,ci,ai,1,sl]=(pd1.exp()*(pd1-pd0)).sum(-1).cpu().numpy();kl[:,ci,ai,2,sl]=(tp[None]*(pt[None]-pd1)).sum(-1).cpu().numpy()
   print('condition',ci,'amplitude',amp,'deleted forward audit',audit,flush=True)
  np.savez_compressed(STEM.with_name(STEM.name+'_partial_rows.npz'),kl=kl[:,:ci+1],native_effect=effect[:ci+1],raw_error=rawerr[:,:ci+1],raw_energy=rawenergy[:ci+1])
 assert np.isfinite(kl).all() and kl.min()>-1e-10 and effect.min()>0
 np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),kl=kl,native_effect=effect,raw_error=rawerr,raw_energy=rawenergy);analyze(cfg,kl,effect,rawerr,rawenergy,intact,{'max_deleted_forward_normalized_state_absolute_error':audit,'geometry_errors':geometry})
def analyze(cfg,kl,effect,rawerr,rawenergy,intact,audit):
 n=cfg['rows'];k=kl.reshape(8,7,2,3,n,4).mean(-1);e=effect.reshape(7,2,n,4).mean(-1);r=rawerr.reshape(8,7,2,n,4).mean(-1);b=rawenergy.reshape(7,2,n,4).mean(-1);ii=intact.reshape(8,3,n,4).mean(-1)
 def primary(ix):
  return np.array([1-r[0,0,0,ix].mean()/b[0,0,ix].mean(),1-k[0,0,0,0,ix].mean()/e[0,0,ix].mean(),k[0,0,0,1,ix].mean()/e[0,0,ix].mean(),ii[0,1,ix].mean()])
 point=primary(np.arange(n));rng=np.random.default_rng(cfg['bootstrap_seed']);boot=np.stack([primary(rng.integers(0,n,n)) for _ in range(cfg['bootstrap_samples'])]);rad=np.quantile(abs(boot-point).max(1),.95);ci=np.stack([point-rad,point+rad],-1);gates={'raw_unit90':bool(ci[0,0]>.9),'native_unit90':bool(ci[1,0]>.9),'remaining_unit20':bool(ci[2,1]<.2),'intact_loss02':bool(ci[3,1]<.02)};km=np.stack([k[:,0].mean(-1),k[:,1:].mean((1,4))],1);em=np.stack([e[0].mean(-1),e[1:].mean((0,2))]);rm=np.stack([r[:,0].mean(-1),r[:,1:].mean((1,3))],1);bm=np.stack([b[0].mean(-1),b[1:].mean((0,2))]);j={'config':cfg,'families':['actual','rotated'],'methods':['contrast64','energy64']+[f'random64_{i}' for i in range(6)],'kl_metrics':['retained_prediction_error','ablated_head_effect','ablated_prediction_error'],'mean_kl':km.tolist(),'mean_native_effect':em.tolist(),'raw_recovery':(1-rm/bm[None]).tolist(),'mean_intact_metrics':ii.mean(-1).tolist(),'primary_names':['actual_unit_raw_recovery','actual_unit_native_KL_recovery','actual_unit_remaining_effect_fraction','intact_next_token_NLL_change'],'primary':point.tolist(),'primary_joint4_ci95':ci.tolist(),'audit':audit,'gates':gates};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');np.savez_compressed(STEM.with_name(STEM.name+'_bootstrap.npz'),primary=boot);print('FINISHED',gates,point,flush=True)
if __name__=='__main__':main()

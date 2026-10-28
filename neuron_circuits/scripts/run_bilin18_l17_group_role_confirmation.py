"""H413 fresh confirmation of H412-selected source-dependent group roles."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources
from run_bilin18_l17_native_sensitivity_factors import drms
from run_bilin18_l17_fisher_ranked_factors import native_drms
STEM=ART/'bilin18_l17_group_role_confirmation'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());op=ART/'bilin18_l17_residual_factor_rescue_frozen_sets.npz';assert hashlib.sha256(op.read_bytes()).hexdigest()==cfg['sets_sha256'];assert hashlib.sha256((ROOT/cfg['corpus']).read_bytes()).hexdigest()==cfg['corpus_sha256'];q=np.load(op);core=q['core'];union=q['sets128'][0];addition=union[64:];assert np.array_equal(core,union[:64]) and not set(core)&set(addition);ids=[torch.tensor(v,device='cuda') for v in [core,addition,union]];model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();pre,heads,_=collect_sources(model,np.load(ROOT/cfg['corpus']),cfg);S=len(pre);L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];Dd=D.double();W=model.lm_head.weight.double();x=F.rms_norm(pre,(1152,));lx=x@L.T;rx=x@R.T;phi0=lx*rx;m0=phi0@D.T;state=pre+m0+model.blocks[17].mlp.Down_bias;base=F.rms_norm(state,(1152,));tm=np.empty((3,6,S));fe=np.empty((3,S));finite=np.empty((3,3,2,S));effect=np.empty((3,S));closure=0.;direct_error=0.;tangent_replay=0.
 def lp(y):return (30*torch.tanh((y.double()@W.T)/30)).log_softmax(-1)
 for hi,h in enumerate(cfg['heads']):
  u=drms(pre,-heads[:,h]);phi=((u@L.T)*rx+lx*(u@R.T)).double();full=phi@Dd.T;c=phi[:,ids[0]]@Dd[:,ids[0]].T;a=phi[:,ids[1]]@Dd[:,ids[1]].T;o=full-c-a;xx=F.rms_norm(pre-heads[:,h],(1152,));phi1=(xx@L.T)*(xx@R.T);r=phi1@D.T-m0;delta=phi1-phi0;kept=torch.stack([delta[:,ix]@D[:,ix].T for ix in ids]);target=F.rms_norm(state+r,(1152,));pred=F.rms_norm(state[None]+kept,(1152,));clamp=F.rms_norm(state[None]+r[None]-kept,(1152,))
  for gi,ix in enumerate(ids):
   patch=phi1[:4].clone();patch[:,ix]=phi0[:4,ix];direct=F.rms_norm(pre[:4]+patch@D.T+model.blocks[17].mlp.Down_bias,(1152,));direct_error=max(direct_error,float((direct-clamp[gi,:4]).abs().max()))
  for off in range(0,S,16):
   sl=slice(off,off+16);s=state[sl].double();th=torch.tanh((base[sl].double()@W.T)/30);cap=1-th.square();p=(30*th).softmax(-1);ts=[]
   for v in [c,a,o]:
    t=(native_drms(s,v[sl])@W.T)*cap;t-=(p*t).sum(-1,keepdim=True);ts.append(t)
   tc,ta,to=ts;t=tc+ta+to;f=(p*t.square()).sum(-1);vals=torch.stack([(p*tc.square()).sum(-1),(p*ta.square()).sum(-1),(p*to.square()).sum(-1),2*(p*tc*ta).sum(-1),2*(p*tc*to).sum(-1),2*(p*ta*to).sum(-1)]);tm[hi,:,sl]=vals.cpu().numpy();fe[hi,sl]=f.cpu().numpy();closure=max(closure,float(((vals.sum(0)-f).abs()/f).max()))
   if off==0:
    direct=(native_drms(s,full[sl])@W.T)*cap;direct-=(p*direct).sum(-1,keepdim=True);tangent_replay=max(tangent_replay,float((((p*(direct-t).square()).sum(-1))/f).max()))
   pt=lp(target[sl]);p0=(30*th).log_softmax(-1);effect[hi,sl]=(pt.exp()*(pt-p0)).sum(-1).cpu().numpy()
   for gi in range(3):
    pp=lp(pred[gi,sl]);pc=lp(clamp[gi,sl]);finite[gi,hi,0,sl]=(pt.exp()*(pt-pp)).sum(-1).cpu().numpy();finite[gi,hi,1,sl]=(pc.exp()*(pc-p0)).sum(-1).cpu().numpy()
  print('head',h,'complete',flush=True)
 assert np.isfinite(tm).all() and np.isfinite(finite).all();audit={'Fisher_energy_identity_max_relative':closure,'direct_clamp_state_max_absolute':direct_error,'direct_full_tangent_relative_squared_error':tangent_replay};STEM.with_name(STEM.name+'_audit.json').write_text(json.dumps(audit,indent=2)+'\n');np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),tangent_metrics=tm,full_energy=fe,finite_metrics=finite,native_effect=effect);analyze(cfg,tm,fe,finite,effect,audit)
def analyze(cfg,tm,fe,m,e,audit):
 n=cfg['rows'];td=tm.reshape(3,6,n,4).mean(-1);fd=fe.reshape(3,n,4).mean(-1);d=m.reshape(3,3,2,n,4).mean(-1);ed=e.reshape(3,n,4).mean(-1)
 def summary(ix):
  cr=td[:,3][:,ix].mean(-1)/fd[:,ix].mean(-1);gain=(d[0,0,0,ix].mean()-d[1,0,0,ix].mean())/ed[0,ix].mean();return np.array([cr[1],cr[2],gain])
 point=summary(np.arange(n));rng=np.random.default_rng(cfg['bootstrap_seed']);boot=np.stack([summary(rng.integers(0,n,n)) for _ in range(cfg['bootstrap_samples'])]);rad=np.quantile(abs(boot-point).max(1),.95);ci=np.stack([point-rad,point+rad],-1);doc=json.loads((ART/'bilin18_group_role_confirmation_document_audit.json').read_text());gates={'document_disjointness':doc['overlap']==0 and doc['independently_rehashed_excluded_documents']==8384,'energy_identity':audit['Fisher_energy_identity_max_relative']<1e-8,'head6_cancellation05':bool(ci[0,1]<-.05),'head8_alignment05':bool(ci[1,0]>.05),'head4_additions_over_core10':bool(ci[2,0]>.1)};j={'config':cfg,'tangent_metric_names':['core_energy','additions_energy','remainder_energy','core_add_cross','core_remainder_cross','add_remainder_cross'],'mean_tangent_metrics':tm.mean(-1).tolist(),'mean_full_energy':fe.mean(-1).tolist(),'finite_methods':['core64','additions64','union128'],'finite_metric_names':['prediction_KL','clamped_remaining_KL'],'mean_finite_metrics':m.mean(-1).tolist(),'mean_native_effect':e.mean(-1).tolist(),'primary_names':['head6_core_add_cross_over_full','head8_core_add_cross_over_full','head4_additions_minus_core_finite_recovery'],'primary':point.tolist(),'primary_joint3_ci95':ci.tolist(),'gates':gates,'audit':audit,'document_audit':doc};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');np.savez_compressed(STEM.with_name(STEM.name+'_bootstrap.npz'),primary=boot);print('FINISHED',gates,point,flush=True)
if __name__=='__main__':main()

"""H415 natural donors for fixed group roles."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources
from run_bilin18_l17_native_sensitivity_factors import drms
from run_bilin18_l17_fisher_ranked_factors import native_drms
from run_bilin18_l8_execution_alignment import angle_matched_payload
STEM=ART/'bilin18_l17_group_role_natural_donors'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());op=ART/'bilin18_l17_residual_factor_rescue_frozen_sets.npz';mp=ROOT/cfg['donor_corpus'];assert hashlib.sha256(op.read_bytes()).hexdigest()==cfg['sets_sha256'];assert hashlib.sha256(mp.read_bytes()).hexdigest()==cfg['donor_sha256'];assert hashlib.sha256((ROOT/cfg['corpus']).read_bytes()).hexdigest()==cfg['corpus_sha256'];q=np.load(op);ids=[torch.tensor(v,device='cuda') for v in [q['core'],q['sets128'][0,64:]]];model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();pre,heads,_=collect_sources(model,np.load(ROOT/cfg['corpus']),cfg);S=len(pre);N=cfg['rows'];L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];Dd=D.double();W=model.lm_head.weight.double();x=F.rms_norm(pre,(1152,));lx=x@L.T;rx=x@R.T;phi0=lx*rx;m0=phi0@D.T;state=pre+m0+model.blocks[17].mlp.Down_bias;base=F.rms_norm(state,(1152,));_,donor_heads,_=collect_sources(model,np.load(mp),dict(cfg,corpus=cfg['donor_corpus']));mappings=np.stack([np.random.default_rng(seed).permutation(N) for seed in cfg['draw_seeds']]);np.savez_compressed(STEM.with_name(STEM.name+'_donor_mappings.npz'),document_permutations=mappings);sources={h:torch.linalg.qr(model.blocks[17].attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q for h in [4,6,8]};tm=np.empty((7,2,4,S));kl=np.empty((7,2,S));effect=np.empty((7,S));geometry={'relative_norm':0.,'normalized_residual_dot':0.,'relative_span_residual':0.,'native_relative_span_residual':0.}
 for h,Q in sources.items():geometry['native_relative_span_residual']=max(geometry['native_relative_span_residual'],float(((heads[:,h]-(heads[:,h]@Q)@Q.T).norm(dim=-1)/heads[:,h].norm(dim=-1)).max()))
 def lp(y):return (30*torch.tanh((y.double()@W.T)/30)).log_softmax(-1)
 for condition in range(7):
  if condition:
   donor=torch.tensor(mappings[condition-1],device='cuda');site=(donor[:,None]*4+torch.arange(4,device='cuda')).reshape(-1)
  for hi,h in enumerate([4,6,8]):
   ref=heads[:,h];Q=sources[h]
   if condition==0:pay=ref
   else:
    direction=donor_heads[site,h]@Q;pay=angle_matched_payload(pre,ref,Q,direction.reshape(S,128).float());rn=ref.double().norm(dim=-1);geometry['relative_norm']=max(geometry['relative_norm'],float(((pay.double().norm(dim=-1)-rn).abs()/rn).max()));geometry['normalized_residual_dot']=max(geometry['normalized_residual_dot'],float(((pre.double()*(pay-ref).double()).sum(-1).abs()/(pre.double().norm(dim=-1)*rn)).max()));geometry['relative_span_residual']=max(geometry['relative_span_residual'],float(((pay-(pay@Q)@Q.T).double().norm(dim=-1)/rn).max()));assert max(geometry.values())<1e-5
   if h==4:
    xx=F.rms_norm(pre-pay,(1152,));phi1=(xx@L.T)*(xx@R.T);r=phi1@D.T-m0;delta=phi1-phi0;kept=torch.stack([delta[:,ix]@D[:,ix].T for ix in ids]);target=F.rms_norm(state+r,(1152,));pred=F.rms_norm(state[None]+kept,(1152,))
    for off in range(0,S,16):
     sl=slice(off,off+16);pt=lp(target[sl]);p0=lp(base[sl]);effect[condition,sl]=(pt.exp()*(pt-p0)).sum(-1).cpu().numpy()
     for gi in range(2):
      pp=lp(pred[gi,sl]);kl[condition,gi,sl]=(pt.exp()*(pt-pp)).sum(-1).cpu().numpy()
   else:
    u=drms(pre,-pay);phi=((u@L.T)*rx+lx*(u@R.T)).double();vs=[phi@Dd.T]+[phi[:,ix]@Dd[:,ix].T for ix in ids]
    for off in range(0,S,16):
     sl=slice(off,off+16);s=state[sl].double();th=torch.tanh((base[sl].double()@W.T)/30);cap=1-th.square();p=(30*th).softmax(-1);ts=[]
     for v in vs:
      t=(native_drms(s,v[sl])@W.T)*cap;t-=(p*t).sum(-1,keepdim=True);ts.append(t)
     t,c,a=ts;tm[condition,hi-1,:,sl]=torch.stack([(p*t.square()).sum(-1),2*(p*c*a).sum(-1),(p*c.square()).sum(-1),(p*a.square()).sum(-1)]).cpu().numpy()
  np.savez_compressed(STEM.with_name(STEM.name+'_partial_rows.npz'),tangent_metrics=tm[:condition+1],head4_prediction_KL=kl[:condition+1],head4_effect=effect[:condition+1]);print('condition',condition,'complete',geometry,flush=True)
 assert np.isfinite(tm).all() and np.isfinite(kl).all();np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),tangent_metrics=tm,head4_prediction_KL=kl,head4_effect=effect);STEM.with_name(STEM.name+'_audit.json').write_text(json.dumps(geometry,indent=2)+'\n');analyze(cfg,tm,kl,effect,geometry)
def analyze(cfg,tm,kl,e,geometry):
 old=np.load(ART/'bilin18_l17_group_role_confirmation_rows.npz');mp=ART/'bilin18_l17_group_role_subspace_controls_rows.npz';assert hashlib.sha256(mp.read_bytes()).hexdigest()==cfg['moment_rows_sha256'];mo=np.load(mp);replay=max(float(abs(tm[0,:,0]-old['full_energy'][1:]).max()),float(abs(tm[0,:,1]-old['tangent_metrics'][1:,3]).max()),float(abs(kl[0]-old['finite_metrics'][:2,0,0]).max()),float(abs(e[0]-old['native_effect'][0]).max()));n=cfg['rows'];t=tm.reshape(7,2,4,n,4).mean(-1);k=kl.reshape(7,2,n,4).mean(-1);ef=e.reshape(7,n,4).mean(-1);mt=mo['tangent_metrics'][7:13].reshape(6,2,4,n,4).mean(-1);mk=mo['head4_prediction_KL'][7:13].reshape(6,2,n,4).mean(-1);me=mo['head4_effect'][7:13].reshape(6,n,4).mean(-1)
 def one(tt,kk,ee,draw,docs):
  v=tt[draw][:,:,:,docs].mean((0,3));q=kk[draw][:,:,docs].mean((0,2));den=ee[draw][:,docs].mean();return np.r_[v[:,1]/v[:,0],(q[0]-q[1])/den]
 def summary(docs,draw,mdraw):
  actual=one(t,k,ef,[0],docs);donor=one(t,k,ef,1+draw,docs);moment=one(mt,mk,me,mdraw,docs);return np.r_[donor,donor-actual,donor-moment],actual,moment
 point,actual,moment=summary(np.arange(n),np.arange(6),np.arange(6));rng=np.random.default_rng(cfg['bootstrap_seed']);boot=np.stack([summary(rng.integers(0,n,n),rng.integers(0,6,6),rng.integers(0,6,6))[0] for _ in range(cfg['bootstrap_samples'])]);rad=np.quantile(abs(boot-point).max(1),.95);ci=np.stack([point-rad,point+rad],-1);doc=json.loads((ART/'bilin18_group_role_natural_donor_document_audit.json').read_text());gates={'disjointness':doc['overlap']==0,'geometry':max(geometry.values())<1e-5,'actual_replay':replay<1e-9,'head6_pairing_loss10':bool(ci[3,0]>.1),'head6_donor_cancellation05':bool(ci[0,1]<-.05),'other_roles_preserved':bool(ci[1,0]>.05 and ci[2,0]>.1),'donor_actual_equivalence05':bool((ci[3:6,0]>-.05).all() and (ci[3:6,1]<.05).all()),'donor_moment_equivalence05':bool((ci[6:9,0]>-.05).all() and (ci[6:9,1]<.05).all())};j={'config':cfg,'role_names':['head6_core_add_cross_over_full','head8_core_add_cross_over_full','head4_additions_minus_core_finite_recovery'],'actual_roles':actual.tolist(),'moment_roles':moment.tolist(),'primary_names':['donor_roles','donor_minus_actual','donor_minus_moment'],'primary':point.tolist(),'primary_joint9_ci95':ci.tolist(),'mean_tangent_metrics':tm.mean(-1).tolist(),'mean_head4_prediction_KL':kl.mean(-1).tolist(),'mean_head4_effect':e.mean(-1).tolist(),'gates':gates,'audit':{'geometry':geometry,'actual_arrays_replay_max_absolute':replay},'document_audit':doc};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');np.savez_compressed(STEM.with_name(STEM.name+'_bootstrap.npz'),primary=boot);print('FINISHED',gates,point,flush=True)
if __name__=='__main__':main()

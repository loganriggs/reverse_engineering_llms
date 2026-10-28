"""H416 raw/matched within-panel derangements."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources
from run_bilin18_l17_native_sensitivity_factors import drms
from run_bilin18_l17_fisher_ranked_factors import native_drms
from run_bilin18_l8_execution_alignment import angle_matched_payload
STEM=ART/'bilin18_l17_group_role_within_panel'
def derangement(n,seed):
 rng=np.random.default_rng(seed)
 while True:
  p=rng.permutation(n)
  if np.all(p!=np.arange(n)):return p
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());op=ART/'bilin18_l17_residual_factor_rescue_frozen_sets.npz';assert hashlib.sha256(op.read_bytes()).hexdigest()==cfg['sets_sha256'];assert hashlib.sha256((ROOT/cfg['corpus']).read_bytes()).hexdigest()==cfg['corpus_sha256'];q=np.load(op);ids=[torch.tensor(v,device='cuda') for v in [q['core'],q['sets128'][0,64:]]];model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();pre,heads,_=collect_sources(model,np.load(ROOT/cfg['corpus']),cfg);S=len(pre);N=cfg['rows'];L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];Dd=D.double();W=model.lm_head.weight.double();x=F.rms_norm(pre,(1152,));lx=x@L.T;rx=x@R.T;phi0=lx*rx;m0=phi0@D.T;state=pre+m0+model.blocks[17].mlp.Down_bias;base=F.rms_norm(state,(1152,));mappings=np.stack([derangement(N,seed) for seed in cfg['draw_seeds']]);np.savez_compressed(STEM.with_name(STEM.name+'_donor_mappings.npz'),document_permutations=mappings);sources={h:torch.linalg.qr(model.blocks[17].attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q for h in [4,6,8]};tm=np.empty((13,2,4,S));kl=np.empty((13,2,S));effect=np.empty((13,S));geometry={'relative_norm':0.,'normalized_residual_dot':0.,'relative_span_residual':0.,'native_relative_span_residual':0.}
 for h,Q in sources.items():geometry['native_relative_span_residual']=max(geometry['native_relative_span_residual'],float(((heads[:,h]-(heads[:,h]@Q)@Q.T).norm(dim=-1)/heads[:,h].norm(dim=-1)).max()))
 def lp(y):return (30*torch.tanh((y.double()@W.T)/30)).log_softmax(-1)
 for condition in range(13):
  if condition:
   donor=torch.tensor(mappings[(condition-1)%6],device='cuda');site=(donor[:,None]*4+torch.arange(4,device='cuda')).reshape(-1)
  for hi,h in enumerate([4,6,8]):
   ref=heads[:,h];Q=sources[h]
   if condition==0:pay=ref
   elif condition<=6:
    pay=heads[site,h];assert torch.equal(pay[torch.argsort(site)],ref)
   else:
    direction=heads[site,h]@Q;pay=angle_matched_payload(pre,ref,Q,direction.reshape(S,128).float());rn=ref.double().norm(dim=-1);geometry['relative_norm']=max(geometry['relative_norm'],float(((pay.double().norm(dim=-1)-rn).abs()/rn).max()));geometry['normalized_residual_dot']=max(geometry['normalized_residual_dot'],float(((pre.double()*(pay-ref).double()).sum(-1).abs()/(pre.double().norm(dim=-1)*rn)).max()));geometry['relative_span_residual']=max(geometry['relative_span_residual'],float(((pay-(pay@Q)@Q.T).double().norm(dim=-1)/rn).max()));assert max(geometry.values())<1e-5
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
 old=np.load(ART/'bilin18_l17_group_role_confirmation_rows.npz');replay=max(float(abs(tm[0,:,0]-old['full_energy'][1:]).max()),float(abs(tm[0,:,1]-old['tangent_metrics'][1:,3]).max()),float(abs(kl[0]-old['finite_metrics'][:2,0,0]).max()),float(abs(e[0]-old['native_effect'][0]).max()));n=cfg['rows'];t=tm.reshape(13,2,4,n,4).mean(-1);k=kl.reshape(13,2,n,4).mean(-1);ef=e.reshape(13,n,4).mean(-1)
 def one(draw,docs):
  v=t[draw][:,:,:,docs].mean((0,3));q=k[draw][:,:,docs].mean((0,2));den=ef[draw][:,docs].mean();return np.r_[v[:,1]/v[:,0],(q[0]-q[1])/den]
 def summary(docs,draw):
  act=one([0],docs);raw=one(1+draw,docs);matched=one(7+draw,docs);return np.r_[raw,matched,raw[0]-act[0],matched[0]-act[0],matched[0]-raw[0]],act
 point,actual=summary(np.arange(n),np.arange(6));rng=np.random.default_rng(cfg['bootstrap_seed']);boot=np.stack([summary(rng.integers(0,n,n),rng.integers(0,6,6))[0] for _ in range(cfg['bootstrap_samples'])]);rad=np.quantile(abs(boot-point).max(1),.95);ci=np.stack([point-rad,point+rad],-1);maps=np.load(STEM.with_name(STEM.name+'_donor_mappings.npz'))['document_permutations'];valid=all(np.array_equal(np.sort(v),np.arange(n)) and np.all(v!=np.arange(n)) for v in maps);gates={'derangements':bool(valid),'geometry':max(geometry.values())<1e-5,'actual_replay':replay<1e-9,'raw_head6_loss10':bool(ci[6,0]>.1),'matched_head6_loss10':bool(ci[7,0]>.1),'head6_matching_equivalence05':bool(ci[8,0]>-.05 and ci[8,1]<.05),'raw_other_roles':bool(ci[1,0]>.05 and ci[2,0]>.1),'matched_other_roles':bool(ci[4,0]>.05 and ci[5,0]>.1)};j={'config':cfg,'role_names':['head6_core_add_cross_over_full','head8_core_add_cross_over_full','head4_additions_minus_core_finite_recovery'],'actual_roles':actual.tolist(),'primary_names':['raw_roles3','matched_roles3','raw_minus_actual_head6','matched_minus_actual_head6','matched_minus_raw_head6'],'primary':point.tolist(),'primary_joint9_ci95':ci.tolist(),'mean_tangent_metrics':tm.mean(-1).tolist(),'mean_head4_prediction_KL':kl.mean(-1).tolist(),'mean_head4_effect':e.mean(-1).tolist(),'gates':gates,'audit':{'geometry':geometry,'actual_arrays_replay_max_absolute':replay,'raw_vector_marginal_bitwise_preserved':True}};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');np.savez_compressed(STEM.with_name(STEM.name+'_bootstrap.npz'),primary=boot);print('FINISHED',gates,point,flush=True)
if __name__=='__main__':main()

"""H417 norm/angle factorial."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources
from run_bilin18_l17_native_sensitivity_factors import drms
from run_bilin18_l17_fisher_ranked_factors import native_drms
from run_bilin18_l8_execution_alignment import angle_matched_payload
STEM=ART/'bilin18_l17_group_role_norm_angle_factorial'
def derangement(n,seed):
 rng=np.random.default_rng(seed)
 while True:
  p=rng.permutation(n)
  if np.all(p!=np.arange(n)):return p
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());op=ART/'bilin18_l17_residual_factor_rescue_frozen_sets.npz';assert hashlib.sha256(op.read_bytes()).hexdigest()==cfg['sets_sha256'];assert hashlib.sha256((ROOT/cfg['corpus']).read_bytes()).hexdigest()==cfg['corpus_sha256'];q=np.load(op);ids=[torch.tensor(v,device='cuda') for v in [q['core'],q['sets128'][0,64:]]];model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();pre,heads,_=collect_sources(model,np.load(ROOT/cfg['corpus']),cfg);S=len(pre);N=cfg['rows'];L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];Dd=D.double();W=model.lm_head.weight.double();x=F.rms_norm(pre,(1152,));lx=x@L.T;rx=x@R.T;phi0=lx*rx;m0=phi0@D.T;state=pre+m0+model.blocks[17].mlp.Down_bias;base=F.rms_norm(state,(1152,));mappings=np.stack([derangement(N,seed) for seed in cfg['draw_seeds']]);np.savez_compressed(STEM.with_name(STEM.name+'_donor_mappings.npz'),document_permutations=mappings);sources={h:torch.linalg.qr(model.blocks[17].attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q for h in [4,6,8]};tm=np.empty((13,2,4,S));kl=np.empty((13,2,S));effect=np.empty((13,S));scales=np.empty((6,3,S));geometry={'relative_norm':0.,'normalized_residual_dot':0.,'relative_span_residual':0.,'native_relative_span_residual':0.}
 for h,Q in sources.items():geometry['native_relative_span_residual']=max(geometry['native_relative_span_residual'],float(((heads[:,h]-(heads[:,h]@Q)@Q.T).norm(dim=-1)/heads[:,h].norm(dim=-1)).max()))
 def lp(y):return (30*torch.tanh((y.double()@W.T)/30)).log_softmax(-1)
 for condition in range(13):
  if condition:
   donor=torch.tensor(mappings[(condition-1)%6],device='cuda');site=(donor[:,None]*4+torch.arange(4,device='cuda')).reshape(-1)
  for hi,h in enumerate([4,6,8]):
   ref=heads[:,h];Q=sources[h]
   if condition==0:pay=ref
   else:
    raw=heads[site,h];rn=ref.norm(dim=-1,keepdim=True);dn=raw.norm(dim=-1,keepdim=True);ratio=rn/dn
    if condition<=6:
     pay=raw*ratio;scales[condition-1,hi]=ratio.squeeze(-1).double().cpu().numpy();target_norm=rn;target_cos=(pre.double()*raw.double()).sum(-1)/(pre.double().norm(dim=-1)*raw.double().norm(dim=-1))
    else:
     angle_ref=ref*(dn/rn);pay=angle_matched_payload(pre,angle_ref,Q,raw@Q);target_norm=dn;target_cos=(pre.double()*ref.double()).sum(-1)/(pre.double().norm(dim=-1)*ref.double().norm(dim=-1))
    geometry['relative_norm']=max(geometry['relative_norm'],float(((pay.double().norm(dim=-1)-target_norm.squeeze(-1).double()).abs()/target_norm.squeeze(-1).double()).max()));cos=(pre.double()*pay.double()).sum(-1)/(pre.double().norm(dim=-1)*pay.double().norm(dim=-1));geometry['normalized_residual_dot']=max(geometry['normalized_residual_dot'],float((cos-target_cos).abs().max()));geometry['relative_span_residual']=max(geometry['relative_span_residual'],float(((pay-(pay@Q)@Q.T).double().norm(dim=-1)/pay.double().norm(dim=-1)).max()));assert max(geometry.values())<1e-5
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
 assert np.isfinite(tm).all() and np.isfinite(kl).all();np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),tangent_metrics=tm,head4_prediction_KL=kl,head4_effect=effect,norm_scales=scales);STEM.with_name(STEM.name+'_audit.json').write_text(json.dumps(geometry,indent=2)+'\n');analyze(cfg,tm,kl,effect,geometry,scales)
def analyze(cfg,tm,kl,e,geometry,scales):
 pp=ART/'bilin18_l17_group_role_within_panel_rows.npz';mp=ART/'bilin18_l17_group_role_within_panel_donor_mappings.npz';assert hashlib.sha256(pp.read_bytes()).hexdigest()==cfg['prior_rows_sha256'];assert hashlib.sha256(mp.read_bytes()).hexdigest()==cfg['prior_maps_sha256'];old=np.load(pp);assert np.array_equal(np.load(mp)['document_permutations'],np.load(STEM.with_name(STEM.name+'_donor_mappings.npz'))['document_permutations']);replay=max(float(abs(tm[0]-old['tangent_metrics'][0]).max()),float(abs(kl[0]-old['head4_prediction_KL'][0]).max()),float(abs(e[0]-old['head4_effect'][0]).max()));expected=old['tangent_metrics'][1:7]*scales[:,1:,None,:]**2;identity=float(np.max(abs(tm[1:7]-expected)/expected[:,:,0,None,:]));ratio_error=float(np.max(abs(tm[1:7,:,1]/tm[1:7,:,0]-old['tangent_metrics'][1:7,:,1]/old['tangent_metrics'][1:7,:,0])));n=cfg['rows'];tt=np.stack([old['tangent_metrics'][1:7],tm[1:7],tm[7:13],old['tangent_metrics'][7:13]]).reshape(4,6,2,4,n,4).mean(-1);kk=np.stack([old['head4_prediction_KL'][1:7],kl[1:7],kl[7:13],old['head4_prediction_KL'][7:13]]).reshape(4,6,2,n,4).mean(-1);ee=np.stack([old['head4_effect'][1:7],e[1:7],e[7:13],old['head4_effect'][7:13]]).reshape(4,6,n,4).mean(-1)
 def summary(docs,draw):
  t=tt[:,draw][:,:,:,:,docs].mean((1,4));k=kk[:,draw][:,:,:,docs].mean((1,3));ef=ee[:,draw][:,:,docs].mean((1,2));roles=np.c_[t[:,:,1]/t[:,:,0],(k[:,0]-k[:,1])/ef];raw,norm,angle,both=roles;contrast=np.stack([norm-raw,angle-raw,both-norm-angle+raw,both-angle,both-norm]);return roles,contrast
 roles,point=summary(np.arange(n),np.arange(6));rng=np.random.default_rng(cfg['bootstrap_seed']);boot=np.stack([summary(rng.integers(0,n,n),rng.integers(0,6,6))[1] for _ in range(cfg['bootstrap_samples'])]);rad=np.quantile(abs(boot[:,:,0]-point[:,0]).max(1),.95);srad=np.quantile(abs(boot[:,:,1:]-point[:,1:]).max((1,2)),.95);ci=np.stack([point[:,0]-rad,point[:,0]+rad],-1);sci=np.stack([point[:,1:]-srad,point[:,1:]+srad],-1);inside=lambda rows:bool((ci[rows,0]>-.05).all() and (ci[rows,1]<.05).all());gates={'geometry':max(geometry.values())<1e-5,'actual_replay':replay<1e-9,'norm_scaling_identity':identity<1e-4 and ratio_error<1e-4,'head6_norm_effect10':bool((ci[[0,3],0]>.1).all()),'head6_angle_effect10':bool((ci[[1,4],0]>.1).all()),'head6_angle_equivalence05':inside([1,4]),'head6_norm_equivalence05':inside([0,3])};j={'config':cfg,'conditions':['raw','norm_only','angle_only','both'],'role_names':['head6_cross_over_full','head8_cross_over_full','head4_additions_minus_core_recovery'],'roles':roles.tolist(),'contrast_names':['norm_minus_raw','angle_minus_raw','factorial_interaction','both_minus_angle','both_minus_norm'],'contrasts':point.tolist(),'head6_joint5_ci95':ci.tolist(),'other_roles_joint10_ci95':sci.tolist(),'gates':gates,'audit':{'geometry':geometry,'actual_replay_max_absolute':replay,'norm_scaling_full_energy_normalized_max_error':identity,'norm_only_site_cross_ratio_max_change':ratio_error}};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');np.savez_compressed(STEM.with_name(STEM.name+'_bootstrap.npz'),contrasts=boot);print('FINISHED',gates,roles,point,flush=True)
if __name__=='__main__':main()

"""H397 raw, normalized and native-output interaction geometry."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l8_execution_alignment import angle_matched_payload
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,B
STEM=ART/'bilin18_l17_composition_stage_geometry'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,d in cfg['source_sha256'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==d
 assert hashlib.sha256((ROOT/cfg['corpus']).read_bytes()).hexdigest()==cfg['corpus_sha256']
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();pre,heads,_=collect_sources(model,np.load(ROOT/cfg['corpus'],mmap_mode='r'),cfg);S=len(pre);N=cfg['rows'];eps=torch.finfo(pre.dtype).eps
 L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];x=F.rms_norm(pre,(1152,));m0=B(x,x,L,R,D);state=pre+m0+model.blocks[17].mlp.Down_bias;baseline=F.rms_norm(state,(1152,));W=model.lm_head.weight.double();s02=pre.square().mean(-1,keepdim=True)+eps
 coeff={h:np.load(ART/f'bilin18_l17_rational_response_identity_h{h}_coefficients.npz') for h in [0,1,6]};mom=np.load(ART/'bilin18_l17_source_moment_controls_moments.npz');means=[torch.tensor(v,device='cuda') for v in mom['means']];roots=[torch.tensor(v,device='cuda') for v in mom['roots']];sources={h:torch.linalg.qr(model.blocks[17].attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q for h in [0,1,6]}
 kl=np.empty((3,13,4,S));base=np.empty_like(kl);balance=np.empty_like(kl);gamma_save=np.empty((3,13,S));relative=np.empty_like(kl);geomerr=0.
 oldrows=np.load(ART/'bilin18_l17_pair_control_composition_rows.npz');errors=np.empty((3,13,4,3,S));effects=np.empty_like(errors);common=np.empty((3,13,4,S));cacheerr=0.
 variants=[('actual',None)]+[('moment',seed) for seed in cfg['draw_seeds']]+[('donor',shift) for shift in cfg['donor_shifts']]
 for ci,(name,seed) in enumerate(variants):
  pays={};co={}
  if name=='moment':noise=torch.randn(9,N,4,128,generator=torch.Generator(device='cuda').manual_seed(seed),device='cuda',dtype=torch.float64)
  for hi in [0,1,6]:
   ref=heads[:,hi]
   if name=='actual':pay=ref
   elif name=='moment':
    direction=torch.einsum('npi,pji->npj',noise[hi],roots[hi])+means[hi];pay=angle_matched_payload(pre,ref,sources[hi],direction.reshape(S,128).float())
   else:
    site=(((torch.arange(N,device='cuda')+seed)%N)[:,None]*4+torch.arange(4,device='cuda')).reshape(-1);pay=angle_matched_payload(pre,ref,sources[hi],heads[site,hi]@sources[hi])
   pays[hi]=pay;co[hi]={k:torch.tensor(coeff[hi][k][ci],device='cuda') for k in coeff[hi].files}
   aa=-(pre*pay).mean(-1,keepdim=True)/s02;bb=pay.square().mean(-1,keepdim=True)/s02;geomerr=max(geomerr,float((aa-co[hi]['alpha']).abs().max()),float((bb-co[hi]['beta']).abs().max()))
  for pi,(hi,ki) in enumerate(cfg['pairs']):
   h=-pays[hi];k=-pays[ki];u=co[hi];v=co[ki];gamma=(h*k).mean(-1,keepdim=True)/s02;gamma_save[pi,ci]=gamma.squeeze(-1).cpu().numpy();c=2*B(h,k,L,R,D)/s02-2*gamma*m0;n2h=u['m2']+2*u['alpha']*u['m1'];n2k=v['m2']+2*v['alpha']*v['m1']
   for ai,(a,b) in enumerate(cfg['amplitude_pairs']):
    nh=a*u['m1']+a*a*n2h;nk=b*v['m1']+b*b*n2k;qh=1+2*a*u['alpha']+a*a*u['beta'];qk=1+2*b*v['alpha']+b*b*v['beta'];q=qh+qk-1+2*a*b*gamma;assert q.min()>0
    rh=nh/qh;rk=nk/qk;pred=rh+rk;full=(nh+nk+a*b*c)/q;xn=F.rms_norm(pre+a*h+b*k,(1152,));native=B(xn,xn,L,R,D)-m0;relative[pi,ci,ai]=((full-native).norm(dim=-1)/native.norm(dim=-1)).cpu().numpy();hn=rh.norm(dim=-1);kn=rk.norm(dim=-1);balance[pi,ci,ai]=(2*torch.minimum(hn,kn)/(hn+kn)).cpu().numpy();target=F.rms_norm(state+native,(1152,));prediction=F.rms_norm(state+pred,(1152,))
    errors[pi,ci,ai,0]=(pred.double()-native.double()).square().sum(-1).cpu().numpy();effects[pi,ci,ai,0]=native.double().square().sum(-1).cpu().numpy();errors[pi,ci,ai,1]=(prediction.double()-target.double()).square().sum(-1).cpu().numpy();effects[pi,ci,ai,1]=(target.double()-baseline.double()).square().sum(-1).cpu().numpy();errors[pi,ci,ai,2]=oldrows['kl'][pi,ci,ai];effects[pi,ci,ai,2]=oldrows['baseline_kl'][pi,ci,ai];common[pi,ci,ai]=errors[pi,ci,ai,0]/state.double().square().sum(-1).cpu().numpy()
    sl=slice(0,4);tl=(30*torch.tanh((target[sl].double()@W.T)/30)).log_softmax(-1);tp=tl.exp();bl=(30*torch.tanh((baseline[sl].double()@W.T)/30)).log_softmax(-1);pl=(30*torch.tanh((prediction[sl].double()@W.T)/30)).log_softmax(-1);ek=(tp*(tl-pl)).sum(-1).cpu().numpy();bk=(tp*(tl-bl)).sum(-1).cpu().numpy();cacheerr=max(cacheerr,float(abs(ek-oldrows['kl'][pi,ci,ai,:4]).max()),float(abs(bk-oldrows['baseline_kl'][pi,ci,ai,:4]).max()))
   print('condition',ci,name,'pair',hi,ki,flush=True)
  np.savez_compressed(STEM.with_name(STEM.name+'_partial_rows.npz'),errors=errors[:,:ci+1],effects=effects[:,:ci+1],common_raw_error=common[:,:ci+1])
 assert np.isfinite(errors).all() and effects.min()>0
 np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),errors=errors,effects=effects,common_raw_error=common)
 analyze(cfg,errors,effects,common,{'cached_kl_max_absolute_replay_error':cacheerr,'saved_geometry_max_absolute_replay':geomerr,'joint_identity_max_float32_relative_error':float(relative.max())})
def analyze(cfg,errors,effects,common,audit):
 n=cfg['rows'];e=errors.reshape(3,13,4,3,n,4).mean(-1);b=effects.reshape(3,13,4,3,n,4).mean(-1);cc=common.reshape(3,13,4,n,4).mean(-1);groups=[slice(0,1),slice(1,7),slice(7,13)];em=np.stack([e[:,g].mean((1,4)) for g in groups],1);bm=np.stack([b[:,g].mean((1,4)) for g in groups],1);ratio=em/bm;cm=np.stack([cc[:,g].mean((1,3)) for g in groups],1);rng=np.random.default_rng(cfg['bootstrap_seed']);br=[];bc=[]
 for _ in range(cfg['bootstrap_samples']):
  ix=rng.integers(0,n,n);rr=[];cr=[]
  for g in groups:
   ee=e[:,g];bb=b[:,g];ce=cc[:,g];draw=rng.integers(0,ee.shape[1],ee.shape[1]);rr.append(ee[:,draw][:,:,:,:,ix].mean((1,4))/bb[:,draw][:,:,:,:,ix].mean((1,4)));cr.append(ce[:,draw][:,:,:,ix].mean((1,3)))
  br.append(np.stack(rr,1));bc.append(np.stack(cr,1))
 br=np.stack(br);bc=np.stack(bc);stage=np.log(ratio[...,1:]/ratio[...,:-1]);bstage=np.log(br[...,1:]/br[...,:-1]);sr=np.quantile(abs(bstage-stage).max((1,2,3,4)),.95);sc=np.stack([stage-sr,stage+sr],-1);raw=ratio[:,:,:2,0];rawbr=br[:,:,:,:2,0];rrad=np.quantile(abs(rawbr-raw).max((1,2,3)),.95);rawci=np.stack([raw-rrad,raw+rrad],-1)
 contrast=np.log(cm[:,1:]/cm[:,0:1]);cb=np.log(bc[:,:,1:]/bc[:,:,0:1]);crad=np.quantile(abs(cb-contrast).max((1,2,3)),.95);cci=np.stack([contrast-crad,contrast+crad],-1)
 gates={'cached_kl_replay':audit['cached_kl_max_absolute_replay_error']<1e-10,'raw_unit99':bool((rawci[...,1]<.01).all()),'normalization_suppresses':bool((sc[:,:,:,0,1]<0).all()),'readout_suppresses':bool((sc[:,:,:,1,1]<0).all())}
 j={'config':cfg,'families':['actual','moment','donor'],'stages':['raw_mlp_squared_norm','normalized_residual_squared_norm','native_kl'],'mean_error':em.tolist(),'mean_effect':bm.tolist(),'relative_error':ratio.tolist(),'log_stage_ratio':stage.tolist(),'log_stage_ratio_joint72_ci95':sc.tolist(),'raw_unit_relative_error_joint18_ci95':rawci.tolist(),'mean_common_scaled_raw_error':cm.tolist(),'log_control_over_actual_common_raw_error':contrast.tolist(),'log_control_over_actual_common_raw_error_joint24_ci95':cci.tolist(),'audit':audit,'gates':gates};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');np.savez_compressed(STEM.with_name(STEM.name+'_bootstrap.npz'),relative_error=br,common_raw_error=bc);print('FINISHED',gates,audit,flush=True)
if __name__=='__main__':main()

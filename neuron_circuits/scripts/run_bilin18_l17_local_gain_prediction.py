"""H401 local gain prediction. No discovery outcome used in selection rules."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l8_execution_alignment import angle_matched_payload
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,B
STEM=ART/'bilin18_l17_local_gain_prediction'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,d in cfg['source_sha256'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==d
 assert hashlib.sha256((ROOT/cfg['corpus']).read_bytes()).hexdigest()==cfg['corpus_sha256']
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();pre,heads,_=collect_sources(model,np.load(ROOT/cfg['corpus'],mmap_mode='r'),cfg);S=len(pre);N=cfg['rows'];eps=torch.finfo(pre.dtype).eps
 L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];x=F.rms_norm(pre,(1152,));m0=B(x,x,L,R,D);state=pre+m0+model.blocks[17].mlp.Down_bias;s02=pre.square().mean(-1,keepdim=True)+eps;scale=state.double().square().sum(-1).cpu().numpy()
 hd0=heads.double();pu=pre.double()/pre.double().norm(dim=-1,keepdim=True);dot=(hd0*pu[:,None]).sum(-1);parallel=dot[:,:,None]*pu[:,None];perp=hd0-parallel;gp=perp@perp.transpose(-1,-2);ev,U=torch.linalg.eigh(gp);root=(U*ev.clamp_min(0).sqrt()[:,None,:])@U.transpose(-1,-2);gram=hd0@hd0.transpose(-1,-2);norm=hd0.norm(dim=-1);geometry=[]
 errors=np.empty((2,7,4,S));effects=np.empty_like(errors);balance=np.empty_like(errors);gamma_save=np.empty((2,7,S));audit=0.;individual=np.empty_like(errors);predicted_energy=np.empty((2,7,4,3,S))
 for ci,seed in enumerate([None]+cfg['draw_seeds']):
  if seed is not None:
   gen=torch.Generator(device='cuda').manual_seed(seed);noise=torch.randn(S,1152,9,generator=gen,device='cuda',dtype=torch.float64);noise=noise-pu[:,:,None]*(pu[:,:,None]*noise).sum(1,keepdim=True);Q,qr=torch.linalg.qr(noise,mode='reduced');sg=torch.sign(torch.diagonal(qr,dim1=-2,dim2=-1));Q=Q*torch.where(sg==0,torch.ones_like(sg),sg)[:,None,:];rotated=((Q@root).transpose(-1,-2)+parallel).float()
   gd=rotated.double()@rotated.double().transpose(-1,-2);dd=(rotated.double()*pu[:,None]).sum(-1);ge=float(((gd-gram).abs()/(norm[:,:,None]*norm[:,None,:])).max());de=float(((dd-dot).abs()/norm).max());geometry.append({'seed':seed,'norm_scaled_gram_max_error':ge,'norm_scaled_residual_dot_max_error':de});assert max(ge,de)<1e-6
  directions=[];coeff=[]
  for h in range(9):
   pay=heads[:,h] if seed is None else rotated[:,h]
   hd=-pay;alpha=(pre*hd).mean(-1,keepdim=True)/s02;beta=hd.square().mean(-1,keepdim=True)/s02;n1=2*B(pre,hd,L,R,D)/s02-2*alpha*m0;n2=B(hd,hd,L,R,D)/s02-beta*m0;directions.append(hd);coeff.append((alpha,beta,n1,n2))
  for pi,(h,k) in enumerate(cfg['pairs']):
   hd= directions[h];kd=directions[k];ah,bh,n1h,n2h=coeff[h];ak,bk,n1k,n2k=coeff[k];gamma=(hd*kd).mean(-1,keepdim=True)/s02;gamma_save[pi,ci]=gamma.squeeze(-1).cpu().numpy();cross=2*B(hd,kd,L,R,D)/s02-2*gamma*m0
   for ai,(a,b) in enumerate(cfg['amplitude_pairs']):
    nh=a*n1h+a*a*n2h;nk=b*n1k+b*b*n2k;qh=1+2*a*ah+a*a*bh;qk=1+2*b*ak+b*b*bk;q=qh+qk-1+2*a*b*gamma;assert q.min()>0
    rh=nh/qh;rk=nk/qk;pred=rh+rk;rational=(nh+nk+a*b*cross)/q;xx=F.rms_norm(pre+a*hd+b*kd,(1152,));target=B(xx,xx,L,R,D)-m0;audit=max(audit,float(((target-rational).norm(dim=-1)/target.norm(dim=-1)).max()));errors[pi,ci,ai]=(pred.double()-target.double()).square().sum(-1).cpu().numpy();effects[pi,ci,ai]=target.double().square().sum(-1).cpu().numpy();individual[pi,ci,ai]=(rh.double().square().sum(-1)+rk.double().square().sum(-1)).cpu().numpy();hn=rh.norm(dim=-1);kn=rk.norm(dim=-1);balance[pi,ci,ai]=(2*torch.minimum(hn,kn)/(hn+kn)).cpu().numpy()
    m2h=n2h-2*ah*n1h;m2k=n2k-2*ak*n1k
    for mi,(ph,pk) in enumerate([(a*n1h,b*n1k),(a*n1h/qh,b*n1k/qk),(a*n1h+a*a*m2h,b*n1k+b*b*m2k)]):predicted_energy[pi,ci,ai,mi]=(ph.double().square().sum(-1)+pk.double().square().sum(-1)).cpu().numpy()
   print('condition',ci,'pair',h,k,'audit',audit,flush=True)
  np.savez_compressed(STEM.with_name(STEM.name+'_partial_rows.npz'),errors=errors[:,:ci+1],effects=effects[:,:ci+1],balance=balance[:,:ci+1],gamma=gamma_save[:,:ci+1],state_squared_norm=scale,individual_energy=individual[:,:ci+1],predicted_energy=predicted_energy[:,:ci+1])
 assert np.isfinite(errors).all() and effects.min()>0
 np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),errors=errors,effects=effects,balance=balance,gamma=gamma_save,state_squared_norm=scale,individual_energy=individual,predicted_energy=predicted_energy)
 analyze(cfg,errors,individual,predicted_energy,scale,audit,geometry)
def analyze(cfg,errors,individual,predicted,scale,audit,geometry):
 n=cfg['rows'];m=np.concatenate([errors[:,:,:,None],individual[:,:,:,None],predicted],axis=3)/scale;m=m.reshape(2,7,4,5,n,4).mean(-1)
 def aggregate(m):
  means=np.stack([m[:,:1].mean((1,4)),m[:,1:].mean((1,4))],1);cal=np.log(means[:,:,:,1:2]/means[:,:,:,2:]);gr=np.log(means[:,0]/means[:,1]);gap=gr[:,:,1:2]-gr[:,:,2:];adj=gr[:,:,0:1]-gr[:,:,2:];return means,cal,gap,adj
 means,cal,gap,adj=aggregate(m);rng=np.random.default_rng(cfg['bootstrap_seed']);bc=[];bg=[];ba=[]
 for _ in range(cfg['bootstrap_samples']):
  ix=rng.integers(0,n,n);draw=np.r_[0,1+rng.integers(0,6,6)];_,c,g,aa=aggregate(m[:,draw][:,:,:,:,ix]);bc.append(c);bg.append(g);ba.append(aa)
 bc=np.stack(bc);bg=np.stack(bg);ba=np.stack(ba)
 def bands(boot,point,axes):
  rad=np.quantile(abs(boot-point).max(axis=axes),.95,axis=0);return np.stack([point-rad,point+rad],-1)
 cc=bands(bc,cal,(1,2,3));gc=bands(bg,gap,(1,2));ac=bands(ba,adj,(1,2));old=np.load(ART/'bilin18_l17_rotation_confirmation_rows.npz');replay=0.
 for pi,ai in [(0,3),(1,2)]:
  replay=max(replay,float(abs(errors[pi,:,ai]-old['errors'][pi,:,0]).max()),float(abs(individual[pi,:,ai]-old['individual_energy'][pi,:,0]).max()))
 def passed(ci,axes):return ((ci[...,0]>np.log(.8))&(ci[...,1]<np.log(1.25))).all(axis=axes).tolist()
 gates={'replay':replay<1e-10,'energy_calibration25':passed(cc,(0,1,2)),'gain_ratio25':passed(gc,(0,1)),'adjusted_error25':passed(ac,(0,1))};j={'config':cfg,'families':['actual','rotated'],'mean_common_scaled_metrics':means.tolist(),'metric_names':['error','finite_energy']+cfg['predictors'],'log_energy_calibration':cal.tolist(),'log_energy_calibration_joint16_per_predictor_ci95':cc.tolist(),'log_gain_ratio_prediction_gap':gap.tolist(),'log_gain_ratio_prediction_gap_joint8_per_predictor_ci95':gc.tolist(),'log_adjusted_error_ratio':adj.tolist(),'log_adjusted_error_ratio_joint8_per_predictor_ci95':ac.tolist(),'audit':{'geometry':geometry,'full_panel_rational_native_relative_error':audit,'H400_replay_max_absolute':replay},'gates':gates};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');np.savez_compressed(STEM.with_name(STEM.name+'_bootstrap.npz'),log_calibration=bc,log_gain_gap=bg,log_adjusted_error=ba);print('FINISHED',gates,flush=True)
if __name__=='__main__':main()

"""H399 geometry-preserving source rotations. No discovery outcome used in selection rules."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l8_execution_alignment import angle_matched_payload
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,B
STEM=ART/'bilin18_l17_gram_preserving_rotation'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,d in cfg['source_sha256'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==d
 assert hashlib.sha256((ROOT/cfg['corpus']).read_bytes()).hexdigest()==cfg['corpus_sha256']
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();pre,heads,_=collect_sources(model,np.load(ROOT/cfg['corpus'],mmap_mode='r'),cfg);S=len(pre);N=cfg['rows'];eps=torch.finfo(pre.dtype).eps
 L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];x=F.rms_norm(pre,(1152,));m0=B(x,x,L,R,D);state=pre+m0+model.blocks[17].mlp.Down_bias;s02=pre.square().mean(-1,keepdim=True)+eps;scale=state.double().square().sum(-1).cpu().numpy()
 hd0=heads.double();pu=pre.double()/pre.double().norm(dim=-1,keepdim=True);dot=(hd0*pu[:,None]).sum(-1);parallel=dot[:,:,None]*pu[:,None];perp=hd0-parallel;gp=perp@perp.transpose(-1,-2);ev,U=torch.linalg.eigh(gp);root=(U*ev.clamp_min(0).sqrt()[:,None,:])@U.transpose(-1,-2);gram=hd0@hd0.transpose(-1,-2);norm=hd0.norm(dim=-1);geometry=[]
 errors=np.empty((36,7,4,S));effects=np.empty_like(errors);balance=np.empty_like(errors);gamma_save=np.empty((36,7,S));audit=0.
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
    rh=nh/qh;rk=nk/qk;pred=rh+rk;target=(nh+nk+a*b*cross)/q;errors[pi,ci,ai]=(pred.double()-target.double()).square().sum(-1).cpu().numpy();effects[pi,ci,ai]=target.double().square().sum(-1).cpu().numpy();hn=rh.norm(dim=-1);kn=rk.norm(dim=-1);balance[pi,ci,ai]=(2*torch.minimum(hn,kn)/(hn+kn)).cpu().numpy()
    xx=F.rms_norm(pre[:4]+a*hd[:4]+b*kd[:4],(1152,));native=B(xx,xx,L,R,D)-m0[:4];audit=max(audit,float(((native-target[:4]).norm(dim=-1)/native.norm(dim=-1)).max()))
   if pi%12==11:print('condition',ci,'pairs complete',pi+1,'audit',audit,flush=True)
  np.savez_compressed(STEM.with_name(STEM.name+'_partial_rows.npz'),errors=errors[:,:ci+1],effects=effects[:,:ci+1],balance=balance[:,:ci+1],gamma=gamma_save[:,:ci+1],state_squared_norm=scale)
 assert np.isfinite(errors).all() and effects.min()>0
 np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),errors=errors,effects=effects,balance=balance,gamma=gamma_save,state_squared_norm=scale)
 old=np.load(ART/'bilin18_l17_all_pair_interaction_screen_rows.npz');replay=max(float(abs(errors[:,0]-old['errors'][:,0]).max()),float(abs(effects[:,0]-old['effects'][:,0]).max()));analyze(cfg,errors,effects,scale,audit,geometry,replay)
def analyze(cfg,errors,effects,scale,audit,geometry,replay):
 n=cfg['rows'];e=errors.reshape(36,7,4,n,4).mean(-1);b=effects.reshape(36,7,4,n,4).mean(-1);c=(errors/scale).reshape(36,7,4,n,4).mean(-1)
 def aggregate(e,b,c):
  rr=np.stack([e[:,:1].mean((1,3))/b[:,:1].mean((1,3)),e[:,1:].mean((1,3))/b[:,1:].mean((1,3))],1);cc=np.stack([c[:,:1].mean((1,3)),c[:,1:].mean((1,3))],1);return rr,cc
 ratio,cm=aggregate(e,b,c);contrast=np.log(cm[:,0]/cm[:,1]);rng=np.random.default_rng(cfg['bootstrap_seed']);br=[];bc=[]
 for _ in range(cfg['bootstrap_samples']):
  ix=rng.integers(0,n,n);draw=np.r_[0,1+rng.integers(0,6,6)];rr,cc=aggregate(e[:,draw][:,:,:,ix],b[:,draw][:,:,:,ix],c[:,draw][:,:,:,ix]);br.append(rr);bc.append(np.log(cc[:,0]/cc[:,1]))
 br=np.stack(br);bc=np.stack(bc);rad=np.quantile(abs(bc-contrast).max((1,2)),.95);ci=np.stack([contrast-rad,contrast+rad],-1);ru=ratio[:,:,:2];bru=br[:,:,:,:2];rrad=np.quantile(abs(bru-ru).max((1,2,3)),.95);rci=np.stack([ru-rrad,ru+rrad],-1);weak=ci[...,1]<np.log(.5);strong=ci[...,0]>np.log(2);selected=[]
 for kind,mask,sign in [('weak',weak,1),('strong',strong,-1)]:
  seen=set();inds=np.argwhere(mask);order=sorted(inds,key=lambda ix:sign*contrast[tuple(ix)])
  for pi,ai in order:
   if int(pi) in seen:continue
   seen.add(int(pi));selected.append({'kind':kind,'pair':cfg['pairs'][int(pi)],'amplitudes':cfg['amplitude_pairs'][int(ai)],'actual_over_control_error':float(np.exp(contrast[pi,ai])),'joint95_ratio_interval':np.exp(ci[pi,ai]).tolist()})
   if len(seen)==2:break
 gates={'geometry':max(max(x['norm_scaled_gram_max_error'],x['norm_scaled_residual_dot_max_error']) for x in geometry)<1e-6,'actual_replay':replay<1e-10,'native_audit':audit<.01,'raw_unit99':bool((rci[...,1]<.01).all()),'weak_candidate':bool(weak.any()),'strong_candidate':bool(strong.any())};j={'config':cfg,'families':['actual','rotated'],'relative_error':ratio.tolist(),'raw_unit_relative_error_joint144_ci95':rci.tolist(),'common_scaled_error':cm.tolist(),'log_actual_over_control_error':contrast.tolist(),'log_actual_over_control_error_joint144_ci95':ci.tolist(),'selected_discovery_candidates':selected,'audit':{'max_float32_direct_native_relative_error':audit,'audit_sites_per_condition_amplitude':4,'geometry':geometry,'actual_replay_max_absolute':replay},'gates':gates};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');np.savez_compressed(STEM.with_name(STEM.name+'_bootstrap.npz'),relative_error=br,log_actual_over_control_error=bc);print('FINISHED',gates,selected,flush=True)
if __name__=='__main__':main()

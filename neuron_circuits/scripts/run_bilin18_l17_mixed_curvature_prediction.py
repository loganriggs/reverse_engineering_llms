"""H402 mixed curvature prediction. No discovery outcome used in selection rules."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l8_execution_alignment import angle_matched_payload
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,B
STEM=ART/'bilin18_l17_mixed_curvature_prediction'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,d in cfg['source_sha256'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==d
 assert hashlib.sha256((ROOT/cfg['corpus']).read_bytes()).hexdigest()==cfg['corpus_sha256']
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();pre,heads,_=collect_sources(model,np.load(ROOT/cfg['corpus'],mmap_mode='r'),cfg);S=len(pre);N=cfg['rows'];eps=torch.finfo(pre.dtype).eps
 L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];x=F.rms_norm(pre,(1152,));m0=B(x,x,L,R,D);state=pre+m0+model.blocks[17].mlp.Down_bias;s02=pre.square().mean(-1,keepdim=True)+eps;scale=state.double().square().sum(-1).cpu().numpy()
 hd0=heads.double();pu=pre.double()/pre.double().norm(dim=-1,keepdim=True);dot=(hd0*pu[:,None]).sum(-1);parallel=dot[:,:,None]*pu[:,None];perp=hd0-parallel;gp=perp@perp.transpose(-1,-2);ev,U=torch.linalg.eigh(gp);root=(U*ev.clamp_min(0).sqrt()[:,None,:])@U.transpose(-1,-2);gram=hd0@hd0.transpose(-1,-2);norm=hd0.norm(dim=-1);geometry=[]
 errors=np.empty((2,7,10,2,S));target_energy=np.empty((2,7,10,S));pred_energy=np.empty_like(errors);linear_energy=np.empty_like(target_energy);cosines=np.empty_like(errors);audit=0.;ld,rd,dweight=L.double(),R.double(),D.double()
 for ci,seed in enumerate([None]+cfg['draw_seeds']):
  if seed is not None:
   gen=torch.Generator(device='cuda').manual_seed(seed);noise=torch.randn(S,1152,9,generator=gen,device='cuda',dtype=torch.float64);noise=noise-pu[:,:,None]*(pu[:,:,None]*noise).sum(1,keepdim=True);Q,qr=torch.linalg.qr(noise,mode='reduced');sg=torch.sign(torch.diagonal(qr,dim1=-2,dim2=-1));Q=Q*torch.where(sg==0,torch.ones_like(sg),sg)[:,None,:];rotated=((Q@root).transpose(-1,-2)+parallel).float()
   gd=rotated.double()@rotated.double().transpose(-1,-2);dd=(rotated.double()*pu[:,None]).sum(-1);ge=float(((gd-gram).abs()/(norm[:,:,None]*norm[:,None,:])).max());de=float(((dd-dot).abs()/norm).max());geometry.append({'seed':seed,'norm_scaled_gram_max_error':ge,'norm_scaled_residual_dot_max_error':de});assert max(ge,de)<1e-6
  directions=[];coeff=[]
  for h in range(9):
   pay=heads[:,h] if seed is None else rotated[:,h]
   hd=-pay;alpha=(pre*hd).mean(-1,keepdim=True)/s02;beta=hd.square().mean(-1,keepdim=True)/s02;n1=2*B(pre,hd,L,R,D)/s02-2*alpha*m0;n2=B(hd,hd,L,R,D)/s02-beta*m0;directions.append(hd);coeff.append((alpha,beta,n1,n2))
  for pi,(h,k) in enumerate(cfg['pairs']):
   hd=directions[h];kd=directions[k];ah,bh,n1h,n2h=[v.double() for v in coeff[h]];ak,bk,n1k,n2k=[v.double() for v in coeff[k]];gamma=((hd*kd).mean(-1,keepdim=True)/s02).double();cross=(2*B(hd,kd,L,R,D)/s02).double()-2*gamma*m0.double();mixed=cross-2*ah*n1k-2*ak*n1h
   pd=pre[:4].double();hh=hd[:4].double();kk=kd[:4].double();xx=F.rms_norm(pd,(1152,),eps=eps);base=B(xx,xx,ld,rd,dweight)
   for ai,(a,b) in enumerate(cfg['amplitude_pairs']):
    qh=1+2*a*ah+a*a*bh;qk=1+2*b*ak+b*b*bk;q=qh+qk-1+2*a*b*gamma;assert q.min()>0
    target=a*b/q*(cross-(n1h+a*n2h)*(2*ak+b*bk+2*a*gamma)/qh-(n1k+b*n2k)*(2*ah+a*bh+2*b*gamma)/qk);target_energy[pi,ci,ai]=target.square().sum(-1).cpu().numpy();linear_energy[pi,ci,ai]=(a*a*n1h.square().sum(-1)+b*b*n1k.square().sum(-1)).cpu().numpy()
    for mi,pred in enumerate([a*b*mixed,a*b*mixed/q]):
     errors[pi,ci,ai,mi]=(pred-target).square().sum(-1).cpu().numpy();pred_energy[pi,ci,ai,mi]=pred.square().sum(-1).cpu().numpy();cosines[pi,ci,ai,mi]=((pred*target).sum(-1)/(pred.norm(dim=-1)*target.norm(dim=-1)).clamp_min(1e-30)).cpu().numpy()
    xj=F.rms_norm(pd+a*hh+b*kk,(1152,),eps=eps);xh=F.rms_norm(pd+a*hh,(1152,),eps=eps);xk=F.rms_norm(pd+b*kk,(1152,),eps=eps);native=B(xj,xj,ld,rd,dweight)-B(xh,xh,ld,rd,dweight)-B(xk,xk,ld,rd,dweight)+base;audit=max(audit,float(((target[:4]-native).norm(dim=-1)/native.norm(dim=-1)).max()))
   print('condition',ci,'pair',h,k,'double native audit',audit,flush=True)
  np.savez_compressed(STEM.with_name(STEM.name+'_partial_rows.npz'),errors=errors[:,:ci+1],target_energy=target_energy[:,:ci+1],predicted_energy=pred_energy[:,:ci+1],linear_energy=linear_energy[:,:ci+1],cosine=cosines[:,:ci+1],state_squared_norm=scale)
 assert np.isfinite(errors).all() and target_energy.min()>0
 np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),errors=errors,target_energy=target_energy,predicted_energy=pred_energy,linear_energy=linear_energy,cosine=cosines,state_squared_norm=scale)
 analyze(cfg,errors,target_energy,pred_energy,linear_energy,scale,audit,geometry)
def analyze(cfg,errors,target,pred,linear,scale,audit,geometry):
 n=cfg['rows'];e=(errors/scale).reshape(2,7,10,2,n,4).mean(-1);t=(target/scale).reshape(2,7,10,n,4).mean(-1);p=(pred/scale).reshape(2,7,10,2,n,4).mean(-1);l=(linear/scale).reshape(2,7,10,n,4).mean(-1)
 def aggregate(e,t,p,l):
  ee=np.stack([e[:,:1].mean((1,4)),e[:,1:].mean((1,4))],1);tt=np.stack([t[:,:1].mean((1,3)),t[:,1:].mean((1,3))],1);pp=np.stack([p[:,:1].mean((1,4)),p[:,1:].mean((1,4))],1);ll=np.stack([l[:,:1].mean((1,3)),l[:,1:].mean((1,3))],1);rec=1-ee/tt[...,None];coupling=(tt[:,0]/ll[:,0])/(tt[:,1]/ll[:,1]);return rec,pp/tt[...,None],coupling
 rec,energy_ratio,coupling=aggregate(e,t,p,l);rng=np.random.default_rng(cfg['bootstrap_seed']);br=[];bc=[]
 for _ in range(cfg['bootstrap_samples']):
  ix=rng.integers(0,n,n);draw=np.r_[0,1+rng.integers(0,6,6)];rr,_,cc=aggregate(e[:,draw][:,:,:,:,ix],t[:,draw][:,:,:,ix],p[:,draw][:,:,:,:,ix],l[:,draw][:,:,:,ix]);br.append(rr);bc.append(np.log(cc))
 br=np.stack(br);bc=np.stack(bc);ci=np.empty(rec.shape+(2,))
 for amp in range(5):
  sl=slice(2*amp,2*amp+2);rad=np.quantile(abs(br[:,:,:,sl]-rec[:,:,sl]).max((1,2,3)),.95,axis=0);ci[:,:,sl]=np.stack([rec[:,:,sl]-rad,rec[:,:,sl]+rad],-1)
 crad=np.quantile(abs(bc-np.log(coupling)).max((1,2)),.95);cci=np.stack([np.log(coupling)-crad,np.log(coupling)+crad],-1)
 gates={'native_double_audit':audit<.001}
 for name,idx,threshold in [('quarter99',0,.99),('unit95',2,.95),('four90',4,.9)]:gates[name]=(ci[:,:,2*idx:2*idx+2,:,0]>threshold).all((0,1,2)).tolist()
 j={'config':cfg,'families':['actual','rotated'],'recovery':rec.tolist(),'recovery_joint8_per_amplitude_predictor_ci95':ci.tolist(),'predicted_over_target_energy':energy_ratio.tolist(),'actual_over_rotated_linear_normalized_coupling':coupling.tolist(),'log_coupling_joint20_ci95':cci.tolist(),'audit':{'max_relative_stable_interaction_vs_native_double':audit,'geometry':geometry,'native_audit_sites_per_condition_amplitude':4},'gates':gates};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');np.savez_compressed(STEM.with_name(STEM.name+'_bootstrap.npz'),recovery=br,log_coupling=bc);print('FINISHED',gates,flush=True)
if __name__=='__main__':main()

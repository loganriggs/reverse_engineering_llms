"""H404 fresh local coupling confirmation. No discovery outcome used in selection rules."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l8_execution_alignment import angle_matched_payload
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,B
STEM=ART/'bilin18_l17_local_coupling_confirmation'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,d in cfg['source_sha256'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==d
 assert hashlib.sha256((ROOT/cfg['corpus']).read_bytes()).hexdigest()==cfg['corpus_sha256']
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();pre,heads,_=collect_sources(model,np.load(ROOT/cfg['corpus'],mmap_mode='r'),cfg);S=len(pre);N=cfg['rows'];eps=torch.finfo(pre.dtype).eps
 L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];x=F.rms_norm(pre,(1152,));m0=B(x,x,L,R,D);state=pre+m0+model.blocks[17].mlp.Down_bias;s02=pre.square().mean(-1,keepdim=True)+eps;scale=state.double().square().sum(-1).cpu().numpy()
 hd0=heads.double();pu=pre.double()/pre.double().norm(dim=-1,keepdim=True);dot=(hd0*pu[:,None]).sum(-1);parallel=dot[:,:,None]*pu[:,None];perp=hd0-parallel;gp=perp@perp.transpose(-1,-2);ev,U=torch.linalg.eigh(gp);root=(U*ev.clamp_min(0).sqrt()[:,None,:])@U.transpose(-1,-2);gram=hd0@hd0.transpose(-1,-2);norm=hd0.norm(dim=-1);geometry=[]
 metrics=np.empty((2,7,7,S));audit=0.;ld,rd,dweight=L.double(),R.double(),D.double()
 for ci,seed in enumerate([None]+cfg['draw_seeds']):
  if seed is not None:
   gen=torch.Generator(device='cuda').manual_seed(seed);noise=torch.randn(S,1152,9,generator=gen,device='cuda',dtype=torch.float64);noise=noise-pu[:,:,None]*(pu[:,:,None]*noise).sum(1,keepdim=True);Q,qr=torch.linalg.qr(noise,mode='reduced');sg=torch.sign(torch.diagonal(qr,dim1=-2,dim2=-1));Q=Q*torch.where(sg==0,torch.ones_like(sg),sg)[:,None,:];rotated=((Q@root).transpose(-1,-2)+parallel).float()
   gd=rotated.double()@rotated.double().transpose(-1,-2);dd=(rotated.double()*pu[:,None]).sum(-1);ge=float(((gd-gram).abs()/(norm[:,:,None]*norm[:,None,:])).max());de=float(((dd-dot).abs()/norm).max());geometry.append({'seed':seed,'norm_scaled_gram_max_error':ge,'norm_scaled_residual_dot_max_error':de});assert max(ge,de)<1e-6
  directions=[];coeff=[]
  for h in range(9):
   pay=heads[:,h] if seed is None else rotated[:,h]
   hd=-pay;alpha=(pre*hd).mean(-1,keepdim=True)/s02;beta=hd.square().mean(-1,keepdim=True)/s02;n1=2*B(pre,hd,L,R,D)/s02-2*alpha*m0;n2=B(hd,hd,L,R,D)/s02-beta*m0;directions.append(hd);coeff.append((alpha,beta,n1,n2))
  for pi,(h,k) in enumerate(cfg['pairs']):
   hd=directions[h];kd=directions[k];ah,bh,n1h,n2h=[v.double() for v in coeff[h]];ak,bk,n1k,n2k=[v.double() for v in coeff[k]];gamma=((hd*kd).mean(-1,keepdim=True)/s02).double();fixed=(2*B(hd,kd,L,R,D)/s02).double();correction=-2*gamma*m0.double()-2*ah*n1k-2*ak*n1h;mixed=fixed+correction
   metrics[pi,ci]=torch.stack([mixed.square().sum(-1),n1h.square().sum(-1)+n1k.square().sum(-1),fixed.square().sum(-1),correction.square().sum(-1),2*(fixed*correction).sum(-1),n1h.square().sum(-1),n1k.square().sum(-1)]).cpu().numpy()
   pd=pre[:4].double();hh=hd[:4].double();kk=kd[:4].double();step=.001;finite=0
   for sh,sk in [(1,1),(1,-1),(-1,1),(-1,-1)]:
    xx=F.rms_norm(pd+step*(sh*hh+sk*kk),(1152,),eps=eps);finite=finite+sh*sk*B(xx,xx,ld,rd,dweight)
   finite=finite/(4*step*step);audit=max(audit,float(((mixed[:4]-finite).norm(dim=-1)/finite.norm(dim=-1)).max()))
  print('condition',ci,'all pairs complete','finite difference audit',audit,flush=True);np.savez_compressed(STEM.with_name(STEM.name+'_partial_rows.npz'),metrics=metrics[:,:ci+1],state_squared_norm=scale)
 assert np.isfinite(metrics).all() and (metrics[:,:,:4]>0).all()
 np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),metrics=metrics,state_squared_norm=scale);analyze(cfg,metrics,scale,audit,geometry)
def analyze(cfg,metrics,scale,audit,geometry):
 n=cfg['rows'];assert (metrics[:,:,5:]>0).all();T=scale*metrics[:,:,0]/(metrics[:,:,5]*metrics[:,:,6]);m=np.concatenate([metrics/scale,T[:,:,None]],axis=2).reshape(2,7,8,n,4).mean(-1)
 def aggregate(m):
  means=np.stack([m[:,:1].mean((1,3)),m[:,1:].mean((1,3))],1);curv=np.log(means[:,0,0]/means[:,1,0]);gain=np.log(means[:,0,1]/means[:,1,1]);norm=np.log(means[:,0,7]/means[:,1,7]);return means,np.stack([curv,gain,curv-gain,norm],-1)
 means,ct=aggregate(m);rng=np.random.default_rng(cfg['bootstrap_seed']);boot=[]
 for _ in range(cfg['bootstrap_samples']):
  ix=rng.integers(0,n,n);draw=np.r_[0,1+rng.integers(0,6,6)];_,c=aggregate(m[:,draw][:,:,:,ix]);boot.append(c)
 boot=np.stack(boot);rad=np.quantile(abs(boot-ct).max(1),.95,axis=0);ci=np.stack([ct-rad,ct+rad],-1);gates={'geometry':max(max(x['norm_scaled_gram_max_error'],x['norm_scaled_residual_dot_max_error']) for x in geometry)<1e-6,'finite_difference':audit<.001,'primary_twofold':bool((ci[:,2,1]<np.log(.5)).all()),'response_normalized_twofold':bool((ci[:,3,1]<np.log(.5)).all())};j={'config':cfg,'families':['actual','rotated'],'metric_names':['mixed_energy','first_order_energy','fixed_scale_energy','normalization_energy','cross_energy','Jh','Jk','response_normalized_T'],'mean_metrics':means.tolist(),'contrast_names':['log_curvature_ratio','log_first_order_ratio','log_primary_coupling_ratio','log_response_normalized_T_ratio'],'log_contrasts':ct.tolist(),'log_contrasts_joint2_per_metric_ci95':ci.tolist(),'audit':{'max_relative_native_double_mixed_FD_error':audit,'geometry':geometry,'minimum_individual_first_order_energy':float(metrics[:,:,5:].min())},'gates':gates};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');np.savez_compressed(STEM.with_name(STEM.name+'_bootstrap.npz'),log_contrasts=boot);print('FINISHED',gates,np.exp(ct).tolist(),flush=True)
if __name__=='__main__':main()

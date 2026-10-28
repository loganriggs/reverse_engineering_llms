"""H405 native factors of the head6 sensitivity advantage; no readout refit."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,B
STEM=ART/'bilin18_l17_native_sensitivity_factors'
def drms(p,h):
 s=(p.square().mean(-1,keepdim=True)+torch.finfo(p.dtype).eps).sqrt();return h/s-p*(p*h).mean(-1,keepdim=True)/s.pow(3)
def mixed_rms(p,h,k):
 s=(p.square().mean(-1,keepdim=True)+torch.finfo(p.dtype).eps).sqrt();ph=(p*h).mean(-1,keepdim=True);pk=(p*k).mean(-1,keepdim=True);hk=(h*k).mean(-1,keepdim=True);return -h*pk/s.pow(3)-k*ph/s.pow(3)-p*hk/s.pow(3)+3*p*ph*pk/s.pow(5)
def rotate(pre,heads,seed):
 if seed is None:return heads
 p=pre.double();u=p/p.norm(dim=-1,keepdim=True);H=heads.double();parallel=(H*u[:,None]).sum(-1,keepdim=True)*u[:,None];V=H-parallel;gram=V@V.transpose(-1,-2);ev,U=torch.linalg.eigh(gram);root=(U*ev.clamp_min(0).sqrt()[:,None,:])@U.transpose(-1,-2);z=torch.randn(len(pre),1152,9,generator=torch.Generator(device='cuda').manual_seed(seed),device='cuda',dtype=torch.float64);z-=u[:,:,None]*(z*u[:,:,None]).sum(1,keepdim=True);Q,R=torch.linalg.qr(z,mode='reduced');sg=torch.sign(torch.diagonal(R,dim1=-2,dim2=-1));Q*=torch.where(sg==0,torch.ones_like(sg),sg)[:,None];return ((Q@root).transpose(-1,-2)+parallel).float()
def features(pre,heads,L,R):
 x=F.rms_norm(pre,(1152,));lx=x@L.T;rx=x@R.T;hs={h:-heads[:,h] for h in [0,5,6]};us={h:drms(pre,hs[h]) for h in hs};ls={h:us[h]@L.T for h in hs};rs={h:us[h]@R.T for h in hs};ph=[ls[h]*rx+lx*rs[h] for h in [0,5,6]]
 for h in [0,5]:
  v=mixed_rms(pre,hs[h],hs[6]);ph.append(ls[h]*rs[6]+ls[6]*rs[h]+(v@L.T)*rx+lx*(v@R.T))
 return torch.stack(ph)
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for key,digest in cfg['corpus_sha256'].items():assert hashlib.sha256((ROOT/cfg[key]).read_bytes()).hexdigest()==digest
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];Dd=D.double();signed=[];energies=[];audit=0.;scoreclosure=0.
 for stage in ['discovery','evaluation']:
  cc=dict(cfg,corpus=cfg[stage+'_corpus']);pre,heads,_=collect_sources(model,np.load(ROOT/cc['corpus'],mmap_mode='r'),cc);S=len(pre);x=F.rms_norm(pre,(1152,));m0=B(x,x,L,R,D);state=pre+m0+model.blocks[17].mlp.Down_bias;scale=state.double().square().sum(-1);seeds=[None]+cfg[stage+'_rotation_seeds']
  if stage=='evaluation':
   rows=np.empty((8,4,7,5,4,S));fullrows=np.empty((7,5,S))
  for ci,seed in enumerate(seeds):
   H=rotate(pre,heads,seed);phi=features(pre,H,L,R);full=phi@D.T;norm=full.double().square().sum(-1);reference=2*B(x,drms(pre,-H[:,6]),L,R,D);audit=max(audit,float(((full[2]-reference).norm(dim=-1)/reference.norm(dim=-1)).max()))
   for ti,h in [(3,0),(4,5)]:
    hdir=-H[:,h];kdir=-H[:,6];u=drms(pre,hdir);v=drms(pre,kdir);uv=mixed_rms(pre,hdir,kdir);reference=2*B(u,v,L,R,D)+2*B(x,uv,L,R,D);audit=max(audit,float(((full[ti]-reference).norm(dim=-1)/reference.norm(dim=-1)).max()))
   if stage=='discovery':
    attribution=phi[2].double()*(full[2].double()@Dd);sc=(attribution/scale[:,None]).mean(0);en=(phi[2].double().square()*Dd.square().sum(0)[None]/scale[:,None]).mean(0);signed.append(sc.cpu().numpy());energies.append(en.cpu().numpy());scoreclosure=max(scoreclosure,abs(float(sc.sum())-float((norm[2]/scale).mean()))/float((norm[2]/scale).mean()))
   else:
    fullrows[ci]=(norm/scale).cpu().numpy()
    for oi,order in enumerate(orders):
     for ri,rank in enumerate(cfg['ranks']):
      ix=torch.tensor(order[:rank],device='cuda');kept=phi[:,:,ix]@D[:,ix].T;omitted=full-kept;ke=kept.double().square().sum(-1);oe=omitted.double().square().sum(-1);cross=2*(kept.double()*omitted.double()).sum(-1);# retained prediction error equals omitted energy
      rows[oi,ri,ci]=torch.stack([ke/scale,oe/scale,cross/scale,(kept.double()-full.double()).square().sum(-1)/scale],1).cpu().numpy()
   print(stage,'condition',ci,'factor audit',audit,flush=True)
  if stage=='discovery':
   signed=np.stack(signed);energies=np.stack(energies);contrast=signed[0]-signed[1:].mean(0);orders=np.stack([np.argsort(-contrast,kind='stable'),np.argsort(-energies[0],kind='stable')]+[np.random.default_rng(s).permutation(4608) for s in cfg['random_seeds']]);np.savez_compressed(STEM.with_name(STEM.name+'_frozen_orders.npz'),orders=orders,signed_attribution=signed,individual_contribution_energy=energies,contrast_score=contrast);print('FROZEN ORDERS',hashlib.sha256(STEM.with_name(STEM.name+'_frozen_orders.npz').read_bytes()).hexdigest(),flush=True)
 np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),metrics=rows,full_energy=fullrows);analyze(cfg,rows,fullrows,{'max_relative_native_factor_replay':audit,'signed_energy_sum_relative_error':scoreclosure})
def analyze(cfg,rows,full,audit):
 n=cfg['rows'];m=rows.reshape(8,4,7,5,4,n,4).mean(-1);f=full.reshape(7,5,n,4).mean(-1)
 def summary(m,f,draw,docs,randoms):
  a=m[:,:,0,:,:][:,:,:,:,docs].mean(-1);c=m[:,:,draw,:,:][:,:,:,:,:,docs].mean((2,5));fa=f[0][:,docs].mean(-1);fc=f[draw][:,:,docs].mean((0,2));adv=fa[2]-fc[2];ret=1-a[:,:,2,3]/fa[2];removed=1-(a[:,:,2,1]-c[:,:,2,1])/adv;return np.array([ret[0,1],removed[0,1],removed[0,1]-removed[randoms,1].mean()])
 point=summary(m,f,np.arange(1,7),np.arange(n),np.arange(2,8));rng=np.random.default_rng(cfg['bootstrap_seed']);boot=[]
 for _ in range(cfg['bootstrap_samples']):boot.append(summary(m,f,1+rng.integers(0,6,6),rng.integers(0,n,n),2+rng.integers(0,6,6)))
 boot=np.stack(boot);rad=np.quantile(abs(boot-point).max(1),.95);ci=np.stack([point-rad,point+rad],-1);means=np.stack([m[:,:,0].mean(-1),m[:,:,1:].mean((2,5))],2);fm=np.stack([f[0].mean(-1),f[1:].mean((0,2))]);gates={'factor_identity':audit['max_relative_native_factor_replay']<1e-4,'contrast64_actual90':bool(ci[0,0]>.9),'contrast64_remove80':bool(ci[1,0]>.8),'contrast64_over_random10':bool(ci[2,0]>.1)};j={'config':cfg,'methods':['contrast','individual_energy']+[f'random{i}' for i in range(6)],'responses':['g0','g5','g6','mixed06','mixed56'],'metrics':['retained_energy','omitted_energy','cross_energy','retained_error'],'families':['actual','rotated'],'mean_metrics':means.tolist(),'mean_full_energy':fm.tolist(),'primary_names':['contrast64_g6_actual_recovery','contrast64_g6_advantage_reduction','contrast64_advantage_reduction_minus_random'],'primary':point.tolist(),'primary_joint3_ci95':ci.tolist(),'audit':audit,'gates':gates};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');np.savez_compressed(STEM.with_name(STEM.name+'_bootstrap.npz'),primary=boot);print('FINISHED',gates,point,flush=True)
if __name__=='__main__':main()

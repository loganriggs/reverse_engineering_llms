"""H395 preregistered two-source composition and mixed-response span audit."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,B
STEM=ART/'bilin18_l17_two_source_composition'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,d in cfg['source_sha256'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==d
 assert hashlib.sha256((ROOT/cfg['corpus']).read_bytes()).hexdigest()==cfg['corpus_sha256']
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float()
 pre,heads,_=collect_sources(model,np.load(ROOT/cfg['corpus'],mmap_mode='r'),cfg);S=len(pre);eps=torch.finfo(pre.dtype).eps
 L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]]
 x=F.rms_norm(pre,(1152,));m0=B(x,x,L,R,D);state=pre+m0+model.blocks[17].mlp.Down_bias;baseline=F.rms_norm(state,(1152,));W=model.lm_head.weight.double();s02=pre.square().mean(-1,keepdim=True)+eps
 coeff={}
 for h in [0,1,6]:
  f=np.load(ART/f'bilin18_l17_rational_response_identity_h{h}_coefficients.npz');coeff[h]={k:torch.tensor(f[k][0],device='cuda') for k in f.files}
 kl=np.empty((3,8,5,S));base=np.empty((3,8,S));qsave=np.empty_like(base);residual=[];doubleerr=0.;floaterr=0.
 for pi,(hi,ki) in enumerate(cfg['pairs']):
  h=-heads[:,hi];k=-heads[:,ki];u=coeff[hi];v=coeff[ki];gamma=(h*k).mean(-1,keepdim=True)/s02
  c=2*B(h,k,L,R,D)/s02-2*gamma*m0;g=-2*gamma*m0
  n2h=u['m2']+2*u['alpha']*u['m1'];n2k=v['m2']+2*v['alpha']*v['m1']
  A=torch.stack([u['m1'],n2h,v['m1'],n2k],-1).double();sol=torch.linalg.lstsq(A,c.double().unsqueeze(-1),driver='gels').solution;cp=(A@sol).squeeze(-1).float();residual.append(((c-cp).norm(dim=-1)/c.norm(dim=-1)).cpu().numpy())
  np.savez_compressed(STEM.with_name(STEM.name+f'_pair{hi}_{ki}_cross.npz'),cross=c.cpu().numpy(),projected_cross=cp.cpu().numpy(),gamma=gamma.cpu().numpy(),projection_coefficients=sol.cpu().numpy())
  for ai,(a,b) in enumerate(cfg['amplitude_pairs']):
   nh=a*u['m1']+a*a*n2h;nk=b*v['m1']+b*b*n2k;qh=1+2*a*u['alpha']+a*a*u['beta'];qk=1+2*b*v['alpha']+b*b*v['beta'];q=qh+qk-1+2*a*b*gamma;qsave[pi,ai]=q.squeeze(-1).cpu().numpy()
   preds=[nh/qh+nk/qk,(nh+nk)/q,(nh+nk+a*b*g)/q,(nh+nk+a*b*cp)/q,(nh+nk+a*b*c)/q]
   xn=F.rms_norm(pre+a*h+b*k,(1152,));native=B(xn,xn,L,R,D)-m0;floaterr=max(floaterr,float(((preds[-1]-native).norm(dim=-1)/native.norm(dim=-1)).max()))
   p=pre[:4].double();hd=h[:4].double();kd=k[:4].double();ld,rd,dd=L.double(),R.double(),D.double();ss=p.square().mean(-1,keepdim=True)+eps;z=p/ss.sqrt();m=B(z,z,ld,rd,dd);ah=(p*hd).mean(-1,keepdim=True)/ss;ak=(p*kd).mean(-1,keepdim=True)/ss;bh=hd.square().mean(-1,keepdim=True)/ss;bk=kd.square().mean(-1,keepdim=True)/ss;gg=(hd*kd).mean(-1,keepdim=True)/ss
   nn=a*(2*B(p,hd,ld,rd,dd)/ss-2*ah*m)+a*a*(B(hd,hd,ld,rd,dd)/ss-bh*m)+b*(2*B(p,kd,ld,rd,dd)/ss-2*ak*m)+b*b*(B(kd,kd,ld,rd,dd)/ss-bk*m)+a*b*(2*B(hd,kd,ld,rd,dd)/ss-2*gg*m)
   dp=nn/(1+2*a*ah+2*b*ak+a*a*bh+b*b*bk+2*a*b*gg);zz=F.rms_norm(p+a*hd+b*kd,(1152,),eps=eps);dn=B(zz,zz,ld,rd,dd)-m;doubleerr=max(doubleerr,float(((dp-dn).norm(dim=-1)/dn.norm(dim=-1)).max()))
   target=F.rms_norm(state+native,(1152,));preds=torch.stack([F.rms_norm(state+t,(1152,)) for t in preds])
   for off in range(0,S,16):
    sl=slice(off,off+16);tl=(30*torch.tanh((target[sl].double()@W.T)/30)).log_softmax(-1);tp=tl.exp();bl=(30*torch.tanh((baseline[sl].double()@W.T)/30)).log_softmax(-1);pl=(30*torch.tanh((preds[:,sl].double()@W.T)/30)).log_softmax(-1);kl[pi,ai,:,sl]=(tp[None]*(tl[None]-pl)).sum(-1).cpu().numpy();base[pi,ai,sl]=(tp*(tl-bl)).sum(-1).cpu().numpy()
   print('pair',hi,ki,'amplitudes',a,b,'double audit',doubleerr,flush=True)
  np.savez_compressed(STEM.with_name(STEM.name+'_partial_rows.npz'),kl=kl[:pi+1],baseline_kl=base[:pi+1],denominator=qsave[:pi+1])
 assert np.isfinite(kl).all() and kl.min()>-1e-10 and base.min()>0 and qsave.min()>0
 np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),kl=kl,baseline_kl=base,denominator=qsave,cross_relative_projection_residual=np.stack(residual))
 e=kl.reshape(3,8,5,512,4).mean(-1);bb=base.reshape(3,8,512,4).mean(-1);rec=1-e.mean(-1)/bb.mean(-1)[:,:,None];rng=np.random.default_rng(cfg['bootstrap_seed']);br=[]
 for _ in range(cfg['bootstrap_samples']):
  ix=rng.integers(0,512,512);br.append(1-e[:,:,:,ix].mean(-1)/bb[:,:,ix].mean(-1)[:,:,None])
 br=np.stack(br);radii=np.quantile(np.abs(br-rec).max(axis=(1,2)),.95,axis=0);ci=np.stack([rec-radii,rec+radii],-1)
 gates={'exact_kl':bool(kl[:,:,4].max()<1e-7),'double_identity':doubleerr<1e-9}
 for mi,name in [(0,'sum_single'),(1,'shared_denominator'),(3,'projected_cross')]:gates[name+'_99']=bool((ci[:,:,mi,0]>.99).all())
 j={'config':cfg,'recovery':rec.tolist(),'recovery_joint24_per_method_ci95':ci.tolist(),'gates':gates,'audit':{'max_native_exact_kl':float(kl[:,:,4].max()),'float64_relative_error':doubleerr,'float32_relative_error':floaterr,'min_denominator':float(qsave.min()),'cross_projection_relative_residual_quantiles':np.quantile(residual,[0,.5,.95,1]).tolist()}}
 STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');np.savez_compressed(STEM.with_name(STEM.name+'_bootstrap.npz'),recovery=br);print('FINISHED',gates,j['audit'],flush=True)
if __name__=='__main__':main()

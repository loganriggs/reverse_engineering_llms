"""H394 rational finite-response identity."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l8_execution_alignment import angle_matched_payload
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,B
def drms(v,d,eps=None):
 scale=(v.square().mean(-1,keepdim=True)+(torch.finfo(v.dtype).eps if eps is None else eps)).sqrt();return d/scale-v*(v*d).mean(-1,keepdim=True)/scale.pow(3)
def rms2(v,h,eps=None):
 scale=(v.square().mean(-1,keepdim=True)+(torch.finfo(v.dtype).eps if eps is None else eps)).sqrt();vh=(v*h).mean(-1,keepdim=True);hh=h.square().mean(-1,keepdim=True);return -h*vh/scale.pow(3)+v*(1.5*vh.square()/scale.pow(5)-.5*hh/scale.pow(3))
STEM=ART/'bilin18_l17_rational_response_identity'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,digest in cfg['source_sha256'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==digest
 assert hashlib.sha256((ROOT/cfg['corpus']).read_bytes()).hexdigest()==cfg['corpus_sha256']
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float()
 pre,heads,_=collect_sources(model,np.load(ROOT/cfg['corpus'],mmap_mode='r'),cfg)
 L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]]
 x=F.rms_norm(pre,(1152,));state=pre+B(x,x,L,R,D)+model.blocks[17].mlp.Down_bias;baseline=F.rms_norm(state,(1152,))
 W=model.lm_head.weight.double();assert model.lm_head.bias is None
 frozen=np.load(ART/'bilin18_l17_probability_axes_finite_frozen_basis.npz');K=torch.tensor(frozen['metric'],device='cuda');bases=[torch.tensor(frozen['eigenvectors'][:,:64],device='cuda'),torch.tensor(np.load(ART/'bilin18_l17_random_span_specificity_seed20270602_basis.npz')['basis'],device='cuda')];S=len(pre);deltas=[]
 N=cfg['rows'];amps=cfg['amplitudes'];kl=np.empty((3,13,len(amps),S));base=np.empty_like(kl);relative=np.empty_like(kl);denoms=np.empty_like(kl);doubleerr=0.;native0=B(x,x,L,R,D);ld,rd,dd=L.double(),R.double(),D.double();eps=torch.finfo(pre.dtype).eps
 moments=np.load(ART/'bilin18_l17_source_moment_controls_moments.npz');means=[torch.tensor(v,device='cuda') for v in moments['means']];roots=[torch.tensor(v,device='cuda') for v in moments['roots']];sources=[torch.linalg.qr(model.blocks[17].attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q for h in range(9)]
 for hi,h in enumerate(cfg['heads']):
  coeff1=[];coeff2=[];alphas=[];betas=[];variants=[('actual',None)]+[(f'moment{i}',seed) for i,seed in enumerate(cfg['draw_seeds'])]+[(f'donor{i}',shift) for i,shift in enumerate(cfg['donor_shifts'])]
  for ci,(name,seed) in enumerate(variants):
   ref=heads[:,h]
   if seed is None:pay=ref
   elif name.startswith('moment'):
    gen=torch.Generator(device='cuda').manual_seed(seed);noise=torch.randn(9,N,4,128,generator=gen,device='cuda',dtype=torch.float64);direction=torch.einsum('npi,pji->npj',noise[h],roots[h])+means[h];pay=angle_matched_payload(pre,ref,sources[h],direction.reshape(S,128).float())
   else:
    site=(((torch.arange(N,device='cuda')+seed)%N)[:,None]*4+torch.arange(4,device='cuda')).reshape(-1);pay=angle_matched_payload(pre,ref,sources[h],heads[site,h]@sources[h])
   u1=drms(pre,-pay);u2=rms2(pre,-pay);m1=B(x,u1,L,R,D)+B(u1,x,L,R,D);m2=B(x,u2,L,R,D)+B(u2,x,L,R,D)+B(u1,u1,L,R,D);s02=pre.square().mean(-1,keepdim=True)+eps;alpha=-(pre*pay).mean(-1,keepdim=True)/s02;beta=pay.square().mean(-1,keepdim=True)/s02;coeff1.append(m1.cpu().numpy());coeff2.append(m2.cpu().numpy());alphas.append(alpha.cpu().numpy());betas.append(beta.cpu().numpy());targets=[];preds=[]
   p=pre[:4].double();hdir=-pay[:4].double();xd=F.rms_norm(p,(1152,),eps=eps);du1=drms(p,hdir,eps);du2=rms2(p,hdir,eps);dm1=2*B(xd,du1,ld,rd,dd);dm2=2*B(xd,du2,ld,rd,dd)+B(du1,du1,ld,rd,dd);ds02=p.square().mean(-1,keepdim=True)+eps;da=(p*hdir).mean(-1,keepdim=True)/ds02;db=hdir.square().mean(-1,keepdim=True)/ds02;dm0=B(xd,xd,ld,rd,dd)
   for ai,amp in enumerate(amps):
    q=1+2*amp*alpha+amp**2*beta;pred=(amp*m1+amp**2*(m2+2*alpha*m1))/q;denoms[hi,ci,ai]=q.squeeze(-1).cpu().numpy();exactx=F.rms_norm(pre-amp*pay,(1152,));native=B(exactx,exactx,L,R,D)-native0;relative[hi,ci,ai]=((pred-native).norm(dim=-1)/native.norm(dim=-1).clamp_min(1e-30)).cpu().numpy();targets.append((F.rms_norm(state+native,(1152,))-baseline).double());preds.append((F.rms_norm(state+pred,(1152,))-baseline).double())
    dx=F.rms_norm(p+amp*hdir,(1152,),eps=eps);dnative=B(dx,dx,ld,rd,dd)-dm0;dpred=(amp*dm1+amp**2*(dm2+2*da*dm1))/(1+2*amp*da+amp**2*db);doubleerr=max(doubleerr,float(((dpred-dnative).norm(dim=-1)/dnative.norm(dim=-1)).max()))
   targets=torch.stack(targets);preds=torch.stack(preds)
   for off in range(0,S,16):
    sl=slice(off,off+16);raw=baseline[sl].double()@W.T;lp0=(30*torch.tanh(raw/30)).log_softmax(-1);truechanges=targets[:,sl]@W.T;changes=preds[:,sl]@W.T;targetlp=(30*torch.tanh((raw[None]+truechanges)/30)).log_softmax(-1);tp=targetlp.exp();predlp=(30*torch.tanh((raw[None]+changes)/30)).log_softmax(-1);kl[hi,ci,:,sl]=(tp*(targetlp-predlp)).sum(-1).cpu().numpy();base[hi,ci,:,sl]=(tp*(targetlp-lp0[None])).sum(-1).cpu().numpy()
   if ci%3==0:print('head',h,'condition',ci,'double identity',doubleerr,flush=True)
  np.savez_compressed(STEM.with_name(STEM.name+f'_h{h}_coefficients.npz'),m1=np.stack(coeff1),m2=np.stack(coeff2),alpha=np.stack(alphas),beta=np.stack(betas))
 assert np.isfinite(kl).all() and kl.min()>-1e-10 and np.isfinite(relative).all() and np.isfinite(denoms).all() and base.min()>0
 np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),kl=kl,baseline_kl=base,mlp_relative_error=relative,denominator=denoms);analyze(cfg,kl,base,{'float64_max_relative_mlp_error':doubleerr,'float32_max_relative_mlp_error':float(relative.max()),'float32_relative_mlp_error_quantiles':np.quantile(relative,[.5,.95,.99,1]).tolist(),'minimum_denominator':float(denoms.min()),'maximum_denominator':float(denoms.max()),'maximum_native_kl':float(kl.max()),'double_audit_sites_per_condition':4})
def analyze(cfg,kl,base,audit):
 n=cfg['rows'];groups=[slice(0,1),slice(1,7),slice(7,13)];errors=kl.reshape(3,13,len(cfg['amplitudes']),n,4);base=base.reshape(errors.shape);rec=np.stack([1-errors[:,g].mean((1,3,4))/base[:,g].mean((1,3,4)) for g in groups],1);rng=np.random.default_rng(cfg['bootstrap_seed']);br=[]
 for _ in range(cfg['bootstrap_samples']):
  ix=rng.integers(0,n,n);rr=[]
  for g in groups:
   e=errors[:,g];b=base[:,g];draw=rng.integers(0,e.shape[1],e.shape[1]);em=e[:,draw][:,:,:,ix].mean((1,3,4));bm=b[:,draw][:,:,:,ix].mean((1,3,4));rr.append(1-em/bm)
  br.append(np.stack(rr,1))
 br=np.stack(br);rad=float(np.quantile(np.abs(br-rec).max(axis=(1,2,3)),.95));ci=np.stack([rec-rad,rec+rad],-1);gates={'native_recovery9999':bool((ci[...,0]>.9999).all()),'native_max_kl':audit['maximum_native_kl']<1e-7,'double_identity':audit['float64_max_relative_mlp_error']<1e-9,'positive_denominator':audit['minimum_denominator']>0};j={'config':cfg,'families':['actual','moment','donor'],'recovery':rec.tolist(),'recovery_joint63_ci95':ci.tolist(),'audit':audit,'gates':gates};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');np.savez_compressed(STEM.with_name(STEM.name+'_bootstrap.npz'),recovery=br);print('FINISHED',gates,'audit',audit,'minimum recovery',rec.min(),flush=True)
if __name__=='__main__':main()

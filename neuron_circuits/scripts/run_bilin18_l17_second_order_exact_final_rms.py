"""H393 second-order MLP with exact final RMS."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l8_execution_alignment import angle_matched_payload
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,B
def drms(v,d):
 scale=(v.square().mean(-1,keepdim=True)+torch.finfo(v.dtype).eps).sqrt();return d/scale-v*(v*d).mean(-1,keepdim=True)/scale.pow(3)
def rms2(v,h):
 scale=(v.square().mean(-1,keepdim=True)+torch.finfo(v.dtype).eps).sqrt();vh=(v*h).mean(-1,keepdim=True);hh=h.square().mean(-1,keepdim=True);return -h*vh/scale.pow(3)+v*(1.5*vh.square()/scale.pow(5)-.5*hh/scale.pow(3))
STEM=ART/'bilin18_l17_second_order_exact_final_rms'
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
 N=cfg['rows'];amps=cfg['amplitudes'];kl=np.empty((3,13,len(amps),3,S));base=np.empty((3,13,len(amps),S));fdaudit=0.;native0=B(x,x,L,R,D);ld,rd,dd=L.double(),R.double(),D.double();eps=torch.finfo(pre.dtype).eps
 moments=np.load(ART/'bilin18_l17_source_moment_controls_moments.npz');means=[torch.tensor(v,device='cuda') for v in moments['means']];roots=[torch.tensor(v,device='cuda') for v in moments['roots']];sources=[torch.linalg.qr(model.blocks[17].attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q for h in range(9)]
 for hi,h in enumerate(cfg['heads']):
  coeff1=[];coeff2=[];variants=[('actual',None)]+[(f'moment{i}',seed) for i,seed in enumerate(cfg['draw_seeds'])]+[(f'donor{i}',shift) for i,shift in enumerate(cfg['donor_shifts'])]
  for ci,(name,seed) in enumerate(variants):
   ref=heads[:,h]
   if seed is None:pay=ref
   elif name.startswith('moment'):
    gen=torch.Generator(device='cuda').manual_seed(seed);noise=torch.randn(9,N,4,128,generator=gen,device='cuda',dtype=torch.float64);direction=torch.einsum('npi,pji->npj',noise[h],roots[h])+means[h];pay=angle_matched_payload(pre,ref,sources[h],direction.reshape(S,128).float())
   else:
    site=(((torch.arange(N,device='cuda')+seed)%N)[:,None]*4+torch.arange(4,device='cuda')).reshape(-1);pay=angle_matched_payload(pre,ref,sources[h],heads[site,h]@sources[h])
   u1=drms(pre,-pay);u2=rms2(pre,-pay);m1=B(x,u1,L,R,D)+B(u1,x,L,R,D);m2=B(x,u2,L,R,D)+B(u2,x,L,R,D)+B(u1,u1,L,R,D);c1=drms(state,m1);c2=drms(state,m2)+rms2(state,m1);coeff1.append(c1.cpu().numpy());coeff2.append(c2.cpu().numpy());targets=[];hybrid=[]
   for amp in amps:
    exactx=F.rms_norm(pre-amp*pay,(1152,));native=B(exactx,exactx,L,R,D)-native0;targets.append((F.rms_norm(state+native,(1152,))-baseline).double());hybrid.append((F.rms_norm(state+amp*m1+amp**2*m2,(1152,))-baseline).double())
   targets=torch.stack(targets);hybrid=torch.stack(hybrid);p=pre[:4].double();pa=pay[:4].double();bias=model.blocks[17].mlp.Down_bias.double()
   def fn(t):
    z=F.rms_norm(p-t*pa,(1152,),eps=eps);return F.rms_norm(p+B(z,z,ld,rd,dd)+bias,(1152,),eps=eps)
   fd=(fn(1e-3)+fn(-1e-3)-2*fn(0))/(2e-6);fdaudit=max(fdaudit,float(((fd-c2[:4].double()).norm(dim=-1)/fd.norm(dim=-1).clamp_min(1e-30)).max()))
   for off in range(0,S,16):
    sl=slice(off,off+16);raw=baseline[sl].double()@W.T;lp0=(30*torch.tanh(raw/30)).log_softmax(-1);first=c1[sl].double()@W.T;second=c2[sl].double()@W.T;truechanges=targets[:,sl]@W.T;hybridchanges=hybrid[:,sl]@W.T
    for ai,amp in enumerate(amps):
     targetlp=(30*torch.tanh((raw+truechanges[ai])/30)).log_softmax(-1);tp=targetlp.exp();base[hi,ci,ai,sl]=(tp*(targetlp-lp0)).sum(-1).cpu().numpy()
     for oi,change in enumerate([amp*first,amp*first+amp**2*second,hybridchanges[ai]]):
      predlp=(30*torch.tanh((raw+change)/30)).log_softmax(-1);kl[hi,ci,ai,oi,sl]=(tp*(targetlp-predlp)).sum(-1).cpu().numpy()
   if ci%3==0:print('head',h,'condition',ci,'second-derivative audit',fdaudit,flush=True)
  print('head completed',h,flush=True)
 old=np.load(ART/'bilin18_l17_amplitude_curvature_factorial_rows.npz');basereplay=float(np.max(np.abs(base-old['baseline_kl'])));linearreplay=float(np.max(np.abs(kl[:,:,:,0]-old['kl'][:,:,:,0])));assert np.isfinite(kl).all() and kl.min()>-1e-10 and basereplay<1e-12 and linearreplay<1e-10 and fdaudit<5e-4
 old2=np.load(ART/'bilin18_l17_second_order_prediction_rows.npz');polyreplay=float(np.max(np.abs(kl[:,:,:,:2]-old2['kl'])));assert polyreplay<1e-10
 np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),kl=kl,baseline_kl=base);analyze(cfg,kl,base,{'h392_both_orders_replay_max_absolute':polyreplay,'baseline_h391_replay_max_absolute':basereplay,'first_order_h391_replay_max_absolute':linearreplay,'second_taylor_coefficient_fd_max_relative':fdaudit,'fd_step':1e-3,'fd_sites_per_condition':4})
def analyze(cfg,kl,base,audit):
 n=cfg['rows'];groups=[slice(0,1),slice(1,7),slice(7,13)];errors=kl.reshape(3,13,4,3,n,4);base=base.reshape(3,13,4,n,4);rec=np.stack([1-errors[:,g].mean((1,4,5))/base[:,g].mean((1,3,4))[:,:,None] for g in groups],1);hybrid=rec[...,2];gain=rec[...,2]-rec[...,1];rng=np.random.default_rng(cfg['bootstrap_seed']);br=[]
 for _ in range(cfg['bootstrap_samples']):
  ix=rng.integers(0,n,n);rr=[]
  for g in groups:
   e=errors[:,g];b=base[:,g];draw=rng.integers(0,e.shape[1],e.shape[1]);em=e[:,draw][:,:,:,:,ix].mean((1,4,5));bm=b[:,draw][:,:,:,ix].mean((1,3,4));rr.append(1-em/bm[:,:,None])
  br.append(np.stack(rr,1))
 br=np.stack(br);bg=br[...,2]-br[...,1];rads=np.quantile(np.abs(br[...,2]-hybrid).max(axis=(1,2)),.95,axis=0);grads=np.quantile(np.abs(bg-gain).max(axis=(1,2)),.95,axis=0);rci=np.stack([hybrid-rads[None,None,:],hybrid+rads[None,None,:]],-1);gci=np.stack([gain-grads[None,None,:],gain+grads[None,None,:]],-1);gates={'hybrid_unit99':bool((rci[:,:,1,0]>.99).all()),'hybrid_two95':bool((rci[:,:,2,0]>.95).all()),'hybrid_four90':bool((rci[:,:,3,0]>.9).all()),'head6_controls_gain10':bool((gci[2,1:,3,0]>.1).all())};j={'config':cfg,'families':['actual','moment','donor'],'order':['first','polynomial_second','second_mlp_exact_final_rms'],'recovery':rec.tolist(),'hybrid_minus_polynomial_second':gain.tolist(),'hybrid_recovery_joint9_per_amplitude_ci95':rci.tolist(),'gain_joint9_per_amplitude_ci95':gci.tolist(),'audit':audit,'gates':gates};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');np.savez_compressed(STEM.with_name(STEM.name+'_bootstrap.npz'),recovery=br,gain=bg);print('FINISHED',gates,'hybrid recovery',hybrid,flush=True)
if __name__=='__main__':main()

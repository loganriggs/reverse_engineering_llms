"""H371 independent-document shared probability geometry confirmation."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l8_execution_alignment import angle_matched_payload
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,B
STEM=ART/'bilin18_l17_shared_geometry_confirmation'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,digest in cfg['source_sha256'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==digest
 assert hashlib.sha256((ROOT/cfg['corpus']).read_bytes()).hexdigest()==cfg['corpus_sha256']
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float()
 pre,heads,_=collect_sources(model,np.load(ROOT/cfg['corpus'],mmap_mode='r'),cfg)
 L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]]
 x=F.rms_norm(pre,(1152,));state=pre+B(x,x,L,R,D)+model.blocks[17].mlp.Down_bias;baseline=F.rms_norm(state,(1152,))
 frozen=np.load(ART/'bilin18_l17_unembedding_rank_prediction_rows.npz');V=torch.tensor(frozen['eigenvectors'][:,:2],device='cuda');W=model.lm_head.weight.double();assert model.lm_head.bias is None
 frozen=np.load(ART/'bilin18_l17_rank_covariance_mixture_frozen_probabilities.npz');shared=torch.tensor(frozen['position_mean'],device='cuda');expected=torch.tensor(frozen['expected_unembedding'],device='cuda').reshape(128,4,1152);assert shared.shape[0]==4 and float((shared.sum(-1)-1).abs().max())<1e-12
 moments=np.load(ART/'bilin18_l17_source_moment_controls_moments.npz');means=[torch.tensor(v,device='cuda') for v in moments['means']];roots=[torch.tensor(v,device='cuda') for v in moments['roots']];sources=[torch.linalg.qr(model.blocks[17].attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q for h in range(9)];native0=B(x,x,L,R,D)
 energy=np.empty((3,13,4,4,512));geometry=0.;identity=0.
 for hi,h in enumerate(cfg['heads']):
  responses=[];variants=[('actual',None)]+[(f'moment{i}',seed) for i,seed in enumerate(cfg['draw_seeds'])]+[(f'donor{i}',shift) for i,shift in enumerate(cfg['donor_shifts'])]
  for ci,(name,seed) in enumerate(variants):
   ref=heads[:,h]
   if seed is None:pay=ref
   elif name.startswith('moment'):
    gen=torch.Generator(device='cuda').manual_seed(seed);noise=torch.randn(9,128,4,128,generator=gen,device='cuda',dtype=torch.float64);direction=torch.einsum('npi,pji->npj',noise[h],roots[h])+means[h];pay=angle_matched_payload(pre,ref,sources[h],direction.reshape(512,128).float())
   else:
    site=(((torch.arange(128,device='cuda')+seed)%128)[:,None]*4+torch.arange(4,device='cuda')).reshape(-1);pay=angle_matched_payload(pre,ref,sources[h],heads[site,h]@sources[h])
   geometry=max(geometry,float(((pay.norm(dim=1)-ref.norm(dim=1)).abs()/ref.norm(dim=1)).max()),float(((pre*(pay-ref)).sum(1).abs()/(pre.norm(dim=1)*ref.norm(dim=1))).max()));response=B(F.rms_norm(pre-pay,(1152,)),F.rms_norm(pre-pay,(1152,)),L,R,D)-native0;responses.append(response.cpu().numpy());target=F.rms_norm(state+response,(1152,));delta=(target-baseline).double();u=(delta@V)@V.T;v=delta-u
   mu=torch.einsum('ipd,jpd->ipj',u.reshape(128,4,1152),expected).reshape(512,128);mv=torch.einsum('ipd,jpd->ipj',v.reshape(128,4,1152),expected).reshape(512,128);mu-=mu.mean(-1,keepdim=True);mv-=mv.mean(-1,keepdim=True);between=torch.stack([mu.square().mean(-1),mv.square().mean(-1),2*(mu*mv).mean(-1),(mu+mv).square().mean(-1)]).cpu().numpy();energy[hi,ci,3]=between
   for off in range(0,512,16):
    sl=slice(off,off+16);du=u[sl]@W.T;dv=v[sl]@W.T;raw=baseline[sl].double()@W.T;own=(30*torch.tanh(raw/30)).softmax(-1)
    for mi,prob in enumerate([own,shared[torch.arange(off,off+16,device='cuda')%4]]):
     ju=du-(du*prob).sum(-1,keepdim=True);jv=dv-(dv*prob).sum(-1,keepdim=True);eu=(prob*ju.square()).sum(-1);ev=(prob*jv.square()).sum(-1);cross=2*(prob*ju*jv).sum(-1);total=(prob*(ju+jv).square()).sum(-1);energy[hi,ci,mi,:,sl]=torch.stack([eu,ev,cross,total]).cpu().numpy();identity=max(identity,float(((eu+ev+cross-total).abs()/total).max()))
   energy[hi,ci,2]=energy[hi,ci,1]-between
  np.savez_compressed(STEM.with_name(STEM.name+f'_h{h}_responses.npz'),responses=np.stack(responses));print('head',h,'complete',flush=True)
 assert np.isfinite(energy).all() and energy[:,:,:,[0,1,3]].min()>-1e-10 and geometry<1e-4 and identity<1e-10
 np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),energy=energy)
 analyze(cfg,energy,{'payload_geometry_max_relative':geometry,'energy_identity_max_relative':identity,'minimum_self_energy':float(energy[:,:,:,[0,1,3]].min())})
def analyze(cfg,energy,audit):
 groups=[slice(0,1),slice(1,7),slice(7,13)];ev=energy.reshape(3,13,4,4,128,4)
 def stats(e):
  z=e[:,:,:,2]/e[:,:,:,3];return z,z[:,:,1]/z[:,:,0]
 em=np.stack([ev[:,g].mean((1,4,5)) for g in groups],1);z,ratio=stats(em);rng=np.random.default_rng(cfg['bootstrap_seed']);bz=[];br=[]
 for _ in range(cfg['bootstrap_samples']):
  ix=rng.integers(0,128,128);ee=[]
  for g in groups:
   e=ev[:,g];rr=rng.integers(0,e.shape[1],e.shape[1]);ee.append(e[:,rr][:,:,:,:,ix].mean((1,4,5)))
  zz,rt=stats(np.stack(ee,1));bz.append(zz[:,:,:3]);br.append(rt)
 def band(b,t):
  b=np.stack(b);rad=float(np.quantile(np.max(np.abs(b-t),axis=tuple(range(1,b.ndim))),.95));return np.stack([t-rad,t+rad],-1)
 zci=band(bz,z[:,:,:3]);rci=band(br,ratio);gates={'fresh_within_cancellation':bool((zci[:,:,2,1]<0).all()),'fresh_own_cancellation':bool((zci[:,:,0,1]<0).all()),'head6_mixture_attenuation':bool((rci[2,1:,1]<1).all())}
 j={'config':cfg,'families':['actual','moment','donor'],'metric_order':['own','frozen_mixture','frozen_within','frozen_between'],'cross_over_total':z.tolist(),'cross_joint27_ci95':zci.tolist(),'mixture_over_own':ratio.tolist(),'mixture_over_own_joint9_ci95':rci.tolist(),'mean_energy':em.tolist(),'audit':audit,'gates':gates};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');np.savez_compressed(STEM.with_name(STEM.name+'_bootstrap.npz'),cross=bz,ratio=br);print('FINISHED',gates,'cross',z,'ratio',ratio,flush=True)
if __name__=='__main__':main()

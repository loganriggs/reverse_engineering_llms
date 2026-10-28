"""H396 actual vs matched control joint intervention composition."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l8_execution_alignment import angle_matched_payload
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,B
STEM=ART/'bilin18_l17_pair_control_composition'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,d in cfg['source_sha256'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==d
 assert hashlib.sha256((ROOT/cfg['corpus']).read_bytes()).hexdigest()==cfg['corpus_sha256']
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();pre,heads,_=collect_sources(model,np.load(ROOT/cfg['corpus'],mmap_mode='r'),cfg);S=len(pre);N=cfg['rows'];eps=torch.finfo(pre.dtype).eps
 L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];x=F.rms_norm(pre,(1152,));m0=B(x,x,L,R,D);state=pre+m0+model.blocks[17].mlp.Down_bias;baseline=F.rms_norm(state,(1152,));W=model.lm_head.weight.double();s02=pre.square().mean(-1,keepdim=True)+eps
 coeff={h:np.load(ART/f'bilin18_l17_rational_response_identity_h{h}_coefficients.npz') for h in [0,1,6]};mom=np.load(ART/'bilin18_l17_source_moment_controls_moments.npz');means=[torch.tensor(v,device='cuda') for v in mom['means']];roots=[torch.tensor(v,device='cuda') for v in mom['roots']];sources={h:torch.linalg.qr(model.blocks[17].attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q for h in [0,1,6]}
 kl=np.empty((3,13,4,S));base=np.empty_like(kl);balance=np.empty_like(kl);gamma_save=np.empty((3,13,S));relative=np.empty_like(kl);geomerr=0.
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
    for off in range(0,S,16):
     sl=slice(off,off+16);tl=(30*torch.tanh((target[sl].double()@W.T)/30)).log_softmax(-1);tp=tl.exp();bl=(30*torch.tanh((baseline[sl].double()@W.T)/30)).log_softmax(-1);pl=(30*torch.tanh((prediction[sl].double()@W.T)/30)).log_softmax(-1);kl[pi,ci,ai,sl]=(tp*(tl-pl)).sum(-1).cpu().numpy();base[pi,ci,ai,sl]=(tp*(tl-bl)).sum(-1).cpu().numpy()
   print('condition',ci,name,'pair',hi,ki,flush=True)
  np.savez_compressed(STEM.with_name(STEM.name+'_partial_rows.npz'),kl=kl[:,:ci+1],baseline_kl=base[:,:ci+1],balance=balance[:,:ci+1],gamma=gamma_save[:,:ci+1],identity_error=relative[:,:ci+1])
 assert np.isfinite(kl).all() and kl.min()>-1e-10 and base.min()>0
 np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),kl=kl,baseline_kl=base,balance=balance,gamma=gamma_save,identity_error=relative)
 old=np.load(ART/'bilin18_l17_two_source_composition_rows.npz');idx=[0,1,4,5];replay=max(float(np.max(abs(kl[:,0]-old['kl'][:,idx,0]))),float(np.max(abs(base[:,0]-old['baseline_kl'][:,idx]))));analyze(cfg,kl,base,{'actual_replay_max_absolute':replay,'full_joint_max_float32_relative_error':float(relative.max()),'saved_single_geometry_max_absolute_error':geomerr})
def analyze(cfg,kl,base,audit):
 n=cfg['rows'];e=kl.reshape(3,13,4,n,4).mean(-1);b=base.reshape(3,13,4,n,4).mean(-1);groups=[slice(0,1),slice(1,7),slice(7,13)];rec=np.stack([1-e[:,g].mean((1,3))/b[:,g].mean((1,3)) for g in groups],1);rng=np.random.default_rng(cfg['bootstrap_seed']);br=[]
 for _ in range(cfg['bootstrap_samples']):
  ix=rng.integers(0,n,n);rr=[]
  for g in groups:
   ee=e[:,g];bb=b[:,g];draw=rng.integers(0,ee.shape[1],ee.shape[1]);rr.append(1-ee[:,draw][:,:,:,ix].mean((1,3))/bb[:,draw][:,:,:,ix].mean((1,3)))
  br.append(np.stack(rr,1))
 br=np.stack(br);rad=np.quantile(abs(br-rec).max((1,2,3)),.95);ci=np.stack([rec-rad,rec+rad],-1);delta=rec[:,0:1]-rec[:,1:];bd=br[:,:,0:1]-br[:,:,1:];drad=np.quantile(abs(bd-delta).max((1,2,3)),.95);dc=np.stack([delta-drad,delta+drad],-1)
 gates={'joint_formula_audit':audit['full_joint_max_float32_relative_error']<.01,'actual_replay':audit['actual_replay_max_absolute']<1e-10,'control_unit99':bool((ci[:,1:,:2,0]>.99).all()),'actual_advantage':bool((dc[...,0]>0).all()),'one_point_equivalence':bool((dc[...,0]>-.01).all() and (dc[...,1]<.01).all())};j={'config':cfg,'families':['actual','moment','donor'],'recovery':rec.tolist(),'recovery_joint36_ci95':ci.tolist(),'actual_minus_controls':delta.tolist(),'actual_minus_controls_joint24_ci95':dc.tolist(),'audit':audit,'gates':gates};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');np.savez_compressed(STEM.with_name(STEM.name+'_bootstrap.npz'),recovery=br);print('FINISHED',gates,audit,flush=True)
if __name__=='__main__':main()

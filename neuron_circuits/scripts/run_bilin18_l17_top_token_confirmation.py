"""H384 fresh top512 confirmation."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l8_execution_alignment import angle_matched_payload
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,B
STEM=ART/'bilin18_l17_top_token_confirmation'
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
 N=cfg['rows'];geometry=0.
 moments=np.load(ART/'bilin18_l17_source_moment_controls_moments.npz');means=[torch.tensor(v,device='cuda') for v in moments['means']];roots=[torch.tensor(v,device='cuda') for v in moments['roots']];sources=[torch.linalg.qr(model.blocks[17].attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q for h in range(9)];native0=B(x,x,L,R,D)
 for hi,h in enumerate(cfg['heads']):
  responses=[];variants=[('actual',None)]+[(f'moment{i}',seed) for i,seed in enumerate(cfg['draw_seeds'])]+[(f'donor{i}',shift) for i,shift in enumerate(cfg['donor_shifts'])]
  for ci,(name,seed) in enumerate(variants):
   ref=heads[:,h]
   if seed is None:pay=ref
   elif name.startswith('moment'):
    gen=torch.Generator(device='cuda').manual_seed(seed);noise=torch.randn(9,N,4,128,generator=gen,device='cuda',dtype=torch.float64);direction=torch.einsum('npi,pji->npj',noise[h],roots[h])+means[h];pay=angle_matched_payload(pre,ref,sources[h],direction.reshape(S,128).float())
   else:
    site=(((torch.arange(N,device='cuda')+seed)%N)[:,None]*4+torch.arange(4,device='cuda')).reshape(-1);pay=angle_matched_payload(pre,ref,sources[h],heads[site,h]@sources[h])
   geometry=max(geometry,float(((pay.norm(dim=1)-ref.norm(dim=1)).abs()/ref.norm(dim=1)).max()),float(((pre*(pay-ref)).sum(1).abs()/(pre.norm(dim=1)*ref.norm(dim=1))).max()));response=B(F.rms_norm(pre-pay,(1152,)),F.rms_norm(pre-pay,(1152,)),L,R,D)-native0;responses.append(response.cpu().numpy());deltas.append((F.rms_norm(state+response,(1152,))-baseline).double())
  np.savez_compressed(STEM.with_name(STEM.name+f'_h{h}_responses.npz'),responses=np.stack(responses));print('head',h,'responses generated',flush=True)
 assert geometry<1e-4
 deltas=torch.stack(deltas);grid=cfg['step_grid'];allbest=[];audits=[];mass=[]
 for bi,V in enumerate(bases):
  mapped=V.T@W.T;kl=np.empty((39,2,2,len(grid),S));stationarity=0.;nonoptimal=0.;ranks=np.empty((2,2,S),dtype=int);masses=np.empty((2,S));norms=np.empty((39,2,2,S))
  for off in range(0,S,8):
   sl=slice(off,off+8);d=deltas[:,sl].transpose(0,1);raw=baseline[sl].double()@W.T;dlog=d@W.T;bp=(30*torch.tanh(raw/30)).softmax(-1);targetlp=(30*torch.tanh((raw[:,None]+dlog)/30)).log_softmax(-1);tp=targetlp.exp();values,indices=bp.topk(512,dim=-1)
   for mi,k in enumerate(cfg['top_k']):
    ids=indices[:,:k] if k else torch.arange(bp.shape[-1],device='cuda')[None].expand(len(bp),-1);p=values[:,:k] if k else bp;masses[mi,sl]=p.sum(-1).cpu().numpy();p=p/p.sum(-1,keepdim=True);jm=mapped.T[ids].transpose(1,2);J=jm-(jm*p[:,None]).sum(-1,keepdim=True);t=dlog.gather(-1,ids[:,None].expand(-1,39,-1));tj=t-(t*p[:,None]).sum(-1,keepdim=True);G=(J*p[:,None])@J.transpose(1,2);rhs=(tj*p[:,None])@J.transpose(1,2)
    for ri,rank in enumerate(cfg['ranks']):
     g=G[:,:rank,:rank];val,vec=torch.linalg.eigh((g+g.transpose(1,2))/2);keep=val>val[:,-1:]*cfg['pinv_rtol'];ranks[mi,ri,sl]=keep.sum(-1).cpu().numpy();iv=torch.where(keep,1/val.clamp_min(1e-300),0);b=rhs[:,:,:rank];coef=((b@vec)*iv[:,None])@vec.transpose(1,2);stationarity=max(stationarity,float(((coef@g-b).norm(dim=-1)/b.norm(dim=-1).clamp_min(1e-30)).max()));eu=d@V[:,:rank];obj=lambda c:((c@g)*c).sum(-1)-2*(c*b).sum(-1);den=(p[:,None]*tj.square()).sum(-1);nonoptimal=max(nonoptimal,float(((obj(coef)-obj(eu))/den.clamp_min(1e-30)).max()));change=coef@mapped[:rank];norms[:,mi,ri,sl]=(coef.square().sum(-1)/d.square().sum(-1).clamp_min(1e-30)).T.cpu().numpy()
     for gi,t in enumerate(grid):
      lp=(30*torch.tanh((raw[:,None]+t*change)/30)).log_softmax(-1);kl[:,mi,ri,gi,sl]=(tp*(targetlp-lp)).sum(-1).T.cpu().numpy()
   if off%512==0:print('basis',bi,'sites',off+8,'of',S,flush=True)
  kl=kl.reshape(3,13,2,2,len(grid),S);best=kl.min(4);assert np.isfinite(kl).all() and kl.min()>-1e-10 and nonoptimal<1e-7 and stationarity<1e-6
  bounds=np.minimum(np.array([W.shape[0],512])[:,None]-1,np.array(cfg['ranks'])[None]);rankok=bool((ranks<=bounds[:,:,None]).all());assert rankok
  np.savez_compressed(STEM.with_name(STEM.name+f'_basis{bi}_rows.npz'),grid_kl=kl,best_kl=best,chosen_index=kl.argmin(4),retained_ranks=ranks,retained_mass=masses,coefficient_norm_ratio=norms);allbest.append(best);audits.append({'basis':bi,'normal_equation_max_relative':stationarity,'max_local_suboptimality':nonoptimal,'minimum_retained_ranks':ranks.min(-1).tolist(),'maximum_retained_ranks':ranks.max(-1).tolist(),'rank_bound':rankok,'coefficient_norm_ratio_max':float(norms.max())});mass.append(masses.mean(-1).tolist())
 analyze(cfg,np.stack(allbest),audits,mass,geometry,kl[:,:,0,0,0])
def analyze(cfg,best,audits,mass,geometry,base):
 n=cfg['rows'];groups=[slice(0,1),slice(1,7),slice(7,13)];errors=best.reshape(2,3,13,2,2,n,4);base=base.reshape(3,13,n,4)
 rec=np.stack([1-errors[:,:,g].mean((2,5,6))/base[:,g].mean((1,2,3))[None,:,None,None] for g in groups],2);gap=rec[:,:,:,0,:]-rec[:,:,:,1,:];rng=np.random.default_rng(cfg['bootstrap_seed']);bg=[]
 for _ in range(cfg['bootstrap_samples']):
  ix=rng.integers(0,n,n);rr=[]
  for g in groups:
   e=errors[:,:,g];b=base[:,g];draw=rng.integers(0,e.shape[2],e.shape[2]);em=e[:,:,draw][:,:,:,:,:,ix].mean((2,5,6));bm=b[:,draw][:,:,ix].mean((1,2,3));rr.append(1-em/bm[None,:,None,None])
  rr=np.stack(rr,2);bg.append(rr[:,:,:,0,:]-rr[:,:,:,1,:])
 bg=np.stack(bg);rad=float(np.quantile(np.max(np.abs(bg-gap),axis=(1,2,3,4)),.95));ci=np.stack([gap-rad,gap+rad],-1);gates={'fresh_top512_within5':bool((ci[:,:,:,:,1]<.05).all()),'rank_bound':all(x['rank_bound'] for x in audits)};j={'config':cfg,'basis_order':['probability','random'],'metric_order':['own_uncapped','top512'],'families':['actual','moment','donor'],'recovery':rec.tolist(),'full_minus_truncated':gap.tolist(),'loss_joint36_ci95':ci.tolist(),'mean_retained_probability_mass':mass,'payload_geometry_max_relative':geometry,'audits':audits,'gates':gates};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');np.savez_compressed(STEM.with_name(STEM.name+'_bootstrap.npz'),loss=bg);print('FINISHED',gates,'mass',mass,'recovery',rec,flush=True)
if __name__=='__main__':main()

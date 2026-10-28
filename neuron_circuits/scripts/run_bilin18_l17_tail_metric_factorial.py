"""H385 tail probability alignment and G/b factorial."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,B
STEM=ART/'bilin18_l17_tail_metric_factorial'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,digest in cfg['source_sha256'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==digest
 assert hashlib.sha256((ROOT/cfg['corpus']).read_bytes()).hexdigest()==cfg['corpus_sha256']
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float()
 pre,_,_=collect_sources(model,np.load(ROOT/cfg['corpus'],mmap_mode='r'),cfg)
 L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]]
 x=F.rms_norm(pre,(1152,));state=pre+B(x,x,L,R,D)+model.blocks[17].mlp.Down_bias;baseline=F.rms_norm(state,(1152,))
 W=model.lm_head.weight.double();assert model.lm_head.bias is None
 frozen=np.load(ART/'bilin18_l17_probability_axes_finite_frozen_basis.npz');K=torch.tensor(frozen['metric'],device='cuda');bases=[torch.tensor(frozen['eigenvectors'][:,:64],device='cuda'),torch.tensor(np.load(ART/'bilin18_l17_random_span_specificity_seed20270602_basis.npz')['basis'],device='cuda')];S=len(pre);deltas=[]
 for h in cfg['heads']:
  responses=np.load(ART/f'bilin18_l17_top_token_confirmation_h{h}_responses.npz')['responses'];deltas.extend([(F.rms_norm(state+torch.tensor(v,device='cuda'),(1152,))-baseline).double() for v in responses])
 deltas=torch.stack(deltas);grid=cfg['step_grid'];allbest=[];audits=[];gen=torch.Generator(device='cpu').manual_seed(cfg['permutation_seed']);permutation=torch.randperm(W.shape[0]-32,generator=gen).to('cuda');np.save(STEM.with_name(STEM.name+'_tail_rank_permutation.npy'),permutation.cpu().numpy())
 for bi,V in enumerate(bases):
  mapped=V.T@W.T;kl=np.empty((39,4,2,len(grid),S));stationarity=0.;minrank=[16,64];norms=np.empty((39,4,2,S));entropyerr=0.;toperr=0.;mass=np.empty(S)
  for off in range(0,S,8):
   sl=slice(off,off+8);d=deltas[:,sl].transpose(0,1);raw=baseline[sl].double()@W.T;dlog=d@W.T;bp=(30*torch.tanh(raw/30)).softmax(-1);values,ids=bp.sort(dim=-1,descending=True,stable=True);sp=bp.clone();sp.scatter_(1,ids[:,32:],values[:,32:][:,permutation]);toperr=max(toperr,float((sp.gather(1,ids[:,:32])-values[:,:32]).abs().max()));entropyerr=max(entropyerr,float(((sp*sp.clamp_min(1e-300).log()).sum(-1)-(bp*bp.clamp_min(1e-300).log()).sum(-1)).abs().max()));mass[sl]=values[:,:32].sum(-1).cpu().numpy();assert torch.allclose(sp.sum(-1),torch.ones(len(sp),device='cuda',dtype=torch.float64),atol=1e-12,rtol=0)
   targetlp=(30*torch.tanh((raw[:,None]+dlog)/30)).log_softmax(-1);tp=targetlp.exp();grams=[];right=[]
   for p in [bp,sp]:
    J=mapped[None]-(mapped[None]*p[:,None]).sum(-1,keepdim=True);tj=dlog-(dlog*p[:,None]).sum(-1,keepdim=True);grams.append((J*p[:,None])@J.transpose(1,2));right.append((tj*p[:,None])@J.transpose(1,2))
   for ri,rank in enumerate(cfg['ranks']):
    for gi,G in enumerate(grams):
     g=G[:,:rank,:rank];val,vec=torch.linalg.eigh((g+g.transpose(1,2))/2);keep=val>val[:,-1:]*cfg['pinv_rtol'];minrank[ri]=min(minrank[ri],int(keep.sum(-1).min()));iv=torch.where(keep,1/val.clamp_min(1e-300),0)
     for bj,Bb in enumerate(right):
      mi=2*gi+bj;b=Bb[:,:,:rank];coef=((b@vec)*iv[:,None])@vec.transpose(1,2);stationarity=max(stationarity,float(((coef@g-b).norm(dim=-1)/b.norm(dim=-1).clamp_min(1e-30)).max()));change=coef@mapped[:rank];norms[:,mi,ri,sl]=(coef.square().sum(-1)/d.square().sum(-1).clamp_min(1e-30)).T.cpu().numpy()
      for ti,t in enumerate(grid):
       lp=(30*torch.tanh((raw[:,None]+t*change)/30)).log_softmax(-1);kl[:,mi,ri,ti,sl]=(tp*(targetlp-lp)).sum(-1).T.cpu().numpy()
   if off%512==0:print('basis',bi,'sites',off+8,'of',S,flush=True)
  kl=kl.reshape(3,13,4,2,len(grid),S);best=kl.min(4);old=np.load(ART/f'bilin18_l17_top_token_confirmation_basis{bi}_rows.npz')['grid_kl'][:,:,0];replay=float(np.max(np.abs(kl[:,:,0]-old)));assert np.isfinite(kl).all() and kl.min()>-1e-10 and stationarity<1e-6 and entropyerr<1e-10 and toperr==0 and replay<1e-9
  np.savez_compressed(STEM.with_name(STEM.name+f'_basis{bi}_rows.npz'),grid_kl=kl,best_kl=best,chosen_index=kl.argmin(4),coefficient_norm_ratio=norms,top32_mass=mass);allbest.append(best);audits.append({'basis':bi,'normal_equation_max_relative':stationarity,'minimum_retained_ranks':minrank,'max_entropy_error':entropyerr,'max_protected_probability_error':toperr,'h384_full_grid_replay_max_absolute':replay,'mean_top32_mass':float(mass.mean()),'max_norm_ratio':float(norms.max())})
 analyze(cfg,np.stack(allbest),audits,kl[:,:,0,0,0])
def analyze(cfg,best,audits,base):
 n=cfg['rows'];groups=[slice(0,1),slice(1,7),slice(7,13)];errors=best.reshape(2,3,13,4,2,n,4);base=base.reshape(3,13,n,4)
 def contrasts(rec):
  oo,os,so,ss=[rec[:,:,:,i] for i in range(4)];geom=.5*((oo-so)+(os-ss));rhs=.5*((oo-os)+(so-ss));return np.stack([oo-ss,geom,rhs,geom-rhs,oo-os-so+ss],3)
 rec=np.stack([1-errors[:,:,g].mean((2,5,6))/base[:,g].mean((1,2,3))[None,:,None,None] for g in groups],2);effect=contrasts(rec);rng=np.random.default_rng(cfg['bootstrap_seed']);be=[]
 for _ in range(cfg['bootstrap_samples']):
  ix=rng.integers(0,n,n);rr=[]
  for g in groups:
   e=errors[:,:,g];b=base[:,g];draw=rng.integers(0,e.shape[2],e.shape[2]);em=e[:,:,draw][:,:,:,:,:,ix].mean((2,5,6));bm=b[:,draw][:,:,ix].mean((1,2,3));rr.append(1-em/bm[None,:,None,None])
  be.append(contrasts(np.stack(rr,2)))
 be=np.stack(be);rad=float(np.quantile(np.max(np.abs(be-effect),axis=(1,2,3,4,5)),.95));ci=np.stack([effect-rad,effect+rad],-1);gates={'tail_alignment5':bool((ci[:,:,:,0,:,0]>.05).all()),'geometry_dominates':bool((ci[:,:,:,3,:,0]>0).all()),'rhs_dominates':bool((ci[:,:,:,3,:,1]<0).all())};j={'config':cfg,'basis_order':['probability','random'],'fit_order':['ownG_ownb','ownG_shuffledb','shuffledG_ownb','shuffledG_shuffledb'],'contrast_order':['own_minus_shuffled','geometry_contribution','rhs_contribution','geometry_minus_rhs','interaction'],'families':['actual','moment','donor'],'recovery':rec.tolist(),'effects':effect.tolist(),'effects_joint180_ci95':ci.tolist(),'audits':audits,'gates':gates};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');np.savez_compressed(STEM.with_name(STEM.name+'_bootstrap.npz'),effects=be);print('FINISHED',gates,'recovery',rec,flush=True)
if __name__=='__main__':main()

"""H382 probability-to-token alignment controls."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,B
STEM=ART/'bilin18_l17_coefficient_probability_alignment'
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
  responses=np.load(ART/f'bilin18_l17_shared_geometry_precision_h{h}_responses.npz')['responses'];deltas.extend([(F.rms_norm(state+torch.tensor(v,device='cuda'),(1152,))-baseline).double() for v in responses])
 deltas=torch.stack(deltas);grid=cfg['step_grid'];allbest=[];audits=[];probs=torch.cat([(30*torch.tanh((baseline[i:i+16].double()@W.T)/30)).softmax(-1) for i in range(0,S,16)]);entropy=-(probs*probs.clamp_min(1e-300).log()).sum(-1);features=torch.stack([entropy,torch.logit(probs.max(-1).values.clamp(1e-10,1-1e-10))],-1).cpu().numpy().reshape(512,4,2);f=(features-features.mean(0))/features.std(0);dist=np.sqrt(((f[:,None]-f[None,:])**2).sum(-1));dist[np.arange(512),np.arange(512)]=np.inf;neighbors=dist.argmin(1);ids=(neighbors*4+np.arange(4)[None]).reshape(-1);assert (ids//4!=np.arange(S)//4).all();matched_distance=dist[np.arange(512)[:,None],neighbors,np.arange(4)[None]];random_distance=dist[np.arange(512),(np.arange(512)+1)%512];reduction=1-float(matched_distance.mean()/random_distance.mean());gen=torch.Generator(device='cpu').manual_seed(cfg['permutation_seed']);permutation=torch.randperm(probs.shape[1],generator=gen).to('cuda');permuted=probs[:,permutation];permutation_entropy_error=float((entropy+(permuted*permuted.clamp_min(1e-300).log()).sum(-1)).abs().max());assert permutation_entropy_error<1e-10
 matching={'mean_standardized_distance':float(matched_distance.mean()),'max_standardized_distance':float(matched_distance.max()),'cyclic_control_mean_distance':float(random_distance.mean()),'relative_distance_reduction':reduction,'matched_top_token_agreement':float((probs.argmax(-1)==probs[torch.tensor(ids,device='cuda')].argmax(-1)).double().mean()),'permutation_entropy_max_error':permutation_entropy_error};np.savez_compressed(STEM.with_name(STEM.name+'_matching.npz'),features=features,donor_site_indices=ids,permutation=permutation.cpu().numpy())
 for bi,V in enumerate(bases):
  mapped=V.T@W.T;kl=np.empty((39,2,2,len(grid),S));stationarity=0.;nonoptimal=0.;minrank=[16,64]
  for off in range(0,S,8):
   sl=slice(off,off+8);d=deltas[:,sl].transpose(0,1);raw=baseline[sl].double()@W.T;dlog=d@W.T;targetlp=(30*torch.tanh((raw[:,None]+dlog)/30)).log_softmax(-1);tp=targetlp.exp()
   for mi,bp in enumerate([probs[torch.tensor(ids[sl],device='cuda')],permuted[sl]]):
    J=mapped[None]-(mapped[None]*bp[:,None]).sum(-1,keepdim=True);G=(J*bp[:,None])@J.transpose(1,2);tj=dlog-(dlog*bp[:,None]).sum(-1,keepdim=True);rhs=(tj*bp[:,None])@J.transpose(1,2)
    for ri,rank in enumerate(cfg['ranks']):
     g=G[:,:rank,:rank];val,vec=torch.linalg.eigh((g+g.transpose(1,2))/2);keep=val>val[:,-1:]*cfg['pinv_rtol'];minrank[ri]=min(minrank[ri],int(keep.sum(-1).min()));iv=torch.where(keep,1/val.clamp_min(1e-300),0);b=rhs[:,:,:rank];coef=((b@vec)*iv[:,None])@vec.transpose(1,2);stationarity=max(stationarity,float(((coef@g-b).norm(dim=-1)/b.norm(dim=-1).clamp_min(1e-30)).max()));eu=d@V[:,:rank];obj=lambda c:((c@g)*c).sum(-1)-2*(c*b).sum(-1);den=(bp[:,None]*tj.square()).sum(-1);nonoptimal=max(nonoptimal,float(((obj(coef)-obj(eu))/den).max()));change=coef@mapped[:rank]
     for gi,t in enumerate(grid):
      lp=(30*torch.tanh((raw[:,None]+t*change)/30)).log_softmax(-1);kl[:,mi,ri,gi,sl]=(tp*(targetlp-lp)).sum(-1).T.cpu().numpy()
   if off%512==0:print('basis',bi,'sites',off+8,'of',S,flush=True)
  kl=kl.reshape(3,13,2,2,len(grid),S);best=kl.min(4);assert np.isfinite(kl).all() and kl.min()>-1e-10 and nonoptimal<1e-7 and stationarity<1e-7
  np.savez_compressed(STEM.with_name(STEM.name+f'_basis{bi}_rows.npz'),grid_kl=kl,best_kl=best,chosen_index=kl.argmin(4));allbest.append(best);audits.append({'basis':bi,'normal_equation_max_relative':stationarity,'max_local_suboptimality':nonoptimal,'minimum_retained_ranks':minrank})
 analyze(cfg,np.stack(allbest),audits,matching)
def analyze(cfg,best,audits,matching):
 n=cfg['rows'];groups=[slice(0,1),slice(1,7),slice(7,13)];own=np.stack([np.load(ART/f'bilin18_l17_shared_metric_coefficients_basis{bi}_rows.npz')['best_kl'][:,:,1] for bi in [0,1]]);errors=np.concatenate([own[:,:,:,None],best],axis=3).reshape(2,3,13,3,2,n,4);base=np.load(ART/'bilin18_l17_probability_axes_oracle_rows.npz')['kl'][:,:,0].reshape(3,13,n,4)
 rec=np.stack([1-errors[:,:,g].mean((2,5,6))/base[:,g].mean((1,2,3))[None,:,None,None] for g in groups],2);gap=rec[:,:,:,0,None,:]-rec[:,:,:,1:,:];rng=np.random.default_rng(cfg['bootstrap_seed']);bg=[]
 for _ in range(cfg['bootstrap_samples']):
  ix=rng.integers(0,n,n);rr=[]
  for g in groups:
   e=errors[:,:,g];b=base[:,g];draw=rng.integers(0,e.shape[2],e.shape[2]);em=e[:,:,draw][:,:,:,:,:,ix].mean((2,5,6));bm=b[:,draw][:,:,ix].mean((1,2,3));rr.append(1-em/bm[None,:,None,None])
  rr=np.stack(rr,2);bg.append(rr[:,:,:,0,None,:]-rr[:,:,:,1:,:])
 bg=np.stack(bg);rad=float(np.quantile(np.max(np.abs(bg-gap),axis=(1,2,3,4,5)),.95));ci=np.stack([gap-rad,gap+rad],-1);gates={'confidence_balance':matching['relative_distance_reduction']>.8,'matched_alignment_gain':bool((ci[:,:,:,0,:,0]>.1).all()),'permuted_alignment_gain':bool((ci[:,:,:,1,:,0]>.1).all())};j={'config':cfg,'basis_order':['probability','random'],'metric_order':['own_uncapped','confidence_matched','spectrum_permuted'],'families':['actual','moment','donor'],'recovery':rec.tolist(),'own_minus_control':gap.tolist(),'gain_joint72_ci95':ci.tolist(),'matching':matching,'audits':audits,'gates':gates};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');np.savez_compressed(STEM.with_name(STEM.name+'_bootstrap.npz'),gain=bg);print('FINISHED',gates,'matching',matching,'recovery',rec,flush=True)
if __name__=='__main__':main()

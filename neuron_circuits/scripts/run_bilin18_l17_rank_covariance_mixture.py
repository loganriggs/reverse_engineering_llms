"""H370 exact within/between covariance accounting."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,B
STEM=ART/'bilin18_l17_rank_covariance_mixture'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,digest in cfg['source_sha256'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==digest
 assert hashlib.sha256((ROOT/cfg['corpus']).read_bytes()).hexdigest()==cfg['corpus_sha256']
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float()
 pre,_,_=collect_sources(model,np.load(ROOT/cfg['corpus'],mmap_mode='r'),cfg)
 L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]]
 x=F.rms_norm(pre,(1152,));state=pre+B(x,x,L,R,D)+model.blocks[17].mlp.Down_bias;baseline=F.rms_norm(state,(1152,))
 frozen=np.load(ART/'bilin18_l17_unembedding_rank_prediction_rows.npz');V=torch.tensor(frozen['eigenvectors'][:,:2],device='cuda');W=model.lm_head.weight.double();assert model.lm_head.bias is None
 probabilities=torch.cat([(30*torch.tanh((baseline[i:i+16].double()@W.T)/30)).softmax(-1) for i in range(0,512,16)])
 expectedW=probabilities@W;indices=np.load(ART/'bilin18_l17_rank_prediction_confirmation_h0_rows.npz')['indices'];old=np.load(ART/'bilin18_l17_rank_cross_probability_transfer_rows.npz')['energy'];energy=np.empty((3,13,2,3,4,512));direct_error=0.
 mask=(torch.arange(128,device='cuda')[None]!=torch.arange(512,device='cuda')[:,None]//4).double();matched_ids=torch.tensor((indices[1:]//4).T,device='cuda')
 def variance(mu,mv,mask):
  den=mask.sum(-1);cu=mu-(mu*mask).sum(-1,keepdim=True)/den[:,None];cv=mv-(mv*mask).sum(-1,keepdim=True)/den[:,None]
  return torch.stack([(cu.square()*mask).sum(-1)/den,(cv.square()*mask).sum(-1)/den,2*(cu*cv*mask).sum(-1)/den,((cu+cv).square()*mask).sum(-1)/den]).cpu().numpy()
 for hi,h in enumerate(cfg['heads']):
  cache=np.load(ART/f'bilin18_l17_rank_prediction_confirmation_h{h}_rows.npz');assert np.array_equal(cache['indices'],indices)
  for ci,response in enumerate(cache['responses']):
   target=F.rms_norm(state+torch.tensor(response,device='cuda',dtype=torch.float32),(1152,));delta=(target-baseline).double();u=(delta@V)@V.T;v=delta-u;mu=torch.einsum('ipd,jpd->ipj',u.reshape(128,4,1152),expectedW.reshape(128,4,1152)).reshape(512,128);mv=torch.einsum('ipd,jpd->ipj',v.reshape(128,4,1152),expectedW.reshape(128,4,1152)).reshape(512,128)
   between=variance(mu,mv,mask);mixture=old[hi,ci,7];energy[hi,ci,1]=np.stack([mixture-between,between,mixture])
   between=variance(torch.gather(mu,1,matched_ids),torch.gather(mv,1,matched_ids),torch.ones((512,6),device='cuda'));within=old[hi,ci,1:7].mean(0);energy[hi,ci,0]=np.stack([within,between,within+between])
   if ci==0:
    ids=torch.tensor(indices[1:,:16],device='cuda');prob=probabilities[ids].mean(0);du=u[:16]@W.T;dv=v[:16]@W.T;du=du-(du*prob).sum(-1,keepdim=True);dv=dv-(dv*prob).sum(-1,keepdim=True);direct=torch.stack([(du.square()*prob).sum(-1),(dv.square()*prob).sum(-1),2*(du*dv*prob).sum(-1),((du+dv).square()*prob).sum(-1)]).cpu().numpy();direct_error=max(direct_error,float(np.max(np.abs(direct-energy[hi,ci,0,2,:,:16]))/np.max(np.abs(direct))))
  print('head',h,'complete',flush=True)
 assert np.isfinite(energy).all() and energy[:,:,:,:,[0,1,3]].min()>-1e-10 and direct_error<1e-10
 identity=float(np.max(np.abs(energy[:,:,:,0]+energy[:,:,:,1]-energy[:,:,:,2]))/np.max(np.abs(energy)));assert identity<1e-12
 np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),energy=energy)
 np.savez_compressed(STEM.with_name(STEM.name+'_frozen_probabilities.npz'),position_mean=probabilities.reshape(128,4,-1).mean(0).cpu().numpy(),expected_unembedding=expectedW.cpu().numpy())
 analyze(cfg,energy,{'matched_direct_mixture_relative_error':direct_error,'decomposition_relative_error':identity,'minimum_self_energy':float(energy[:,:,:,:,[0,1,3]].min())})
def analyze(cfg,energy,audit):
 groups=[slice(0,1),slice(1,7),slice(7,13)];ev=energy.reshape(3,13,2,3,4,128,4)
 def stats(e):
  z=e[:,:,:,:,2]/e[:,:,:,:,3];difference=z[:,:,:,2]-z[:,:,:,0];fraction=e[:,:,:,1,3]/e[:,:,:,2,3];return z,difference,fraction
 em=np.stack([ev[:,g].mean((1,5,6)) for g in groups],1);z,diff,fraction=stats(em);rng=np.random.default_rng(cfg['bootstrap_seed']);bz=[];bd=[];bf=[]
 for _ in range(cfg['bootstrap_samples']):
  ix=rng.integers(0,128,128);ee=[]
  for g in groups:
   e=ev[:,g];rr=rng.integers(0,e.shape[1],e.shape[1]);ee.append(e[:,rr][:,:,:,:,:,ix].mean((1,5,6)))
  zz,dd,ff=stats(np.stack(ee,1));bz.append(zz[:,:,:,0]);bd.append(dd);bf.append(ff)
 def band(b,t):
  b=np.stack(b);rad=float(np.quantile(np.max(np.abs(b-t),axis=tuple(range(1,b.ndim))),.95));return np.stack([t-rad,t+rad],-1)
 zci=band(bz,z[:,:,:,0]);dci=band(bd,diff);fci=band(bf,fraction);gates={'within_cancellation':bool((zci[:,:,:,1]<0).all()),'averaging_equivalent':bool((dci[:,:,:,0]>=-.1).all() and (dci[:,:,:,1]<=.1).all()),'between_energy_majority':bool((fci[:,:,:,0]>.5).all())}
 j={'config':cfg,'families':['actual','moment','donor'],'pools':['six_matched','all127'],'parts':['within','between','mixture'],'cross_over_total':z.tolist(),'within_joint18_ci95':zci.tolist(),'mixture_minus_within':diff.tolist(),'mixture_minus_within_joint18_ci95':dci.tolist(),'between_energy_fraction':fraction.tolist(),'between_energy_fraction_joint18_ci95':fci.tolist(),'mean_energy':em.tolist(),'audit':audit,'gates':gates};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');np.savez_compressed(STEM.with_name(STEM.name+'_bootstrap.npz'),within=bz,difference=bd,fraction=bf);print('FINISHED',gates,'difference',diff,'between_fraction',fraction,flush=True)
if __name__=='__main__':main()

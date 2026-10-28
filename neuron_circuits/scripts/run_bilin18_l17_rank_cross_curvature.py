"""H367 probability-metric cross terms and finite curvature."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,B
STEM=ART/'bilin18_l17_rank_cross_curvature'
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
 kl=np.empty((3,13,3,3,512));energy=np.empty((3,13,4,512));replay=0.;orth=0.
 old=np.load(ART/'bilin18_l17_rank_finite_intervention_rows.npz')['kl']
 for hi,h in enumerate(cfg['heads']):
  responses=np.load(ART/f'bilin18_l17_rank_prediction_confirmation_h{h}_rows.npz')['responses']
  for ci,response in enumerate(responses):
   target=F.rms_norm(state+torch.tensor(response,device='cuda',dtype=torch.float32),(1152,));delta=(target-baseline).double();u=(delta@V)@V.T;v=delta-u;orth=max(orth,float(((u*v).sum(-1).abs()/(u.norm(dim=-1)*v.norm(dim=-1))).max()))
   for off in range(0,512,16):
    sl=slice(off,off+16);raw=baseline[sl].double()@W.T;du=u[sl]@W.T;dv=v[sl]@W.T;prob=(30*torch.tanh(raw/30)).softmax(-1);cap=1-torch.tanh(raw/30).square();ju=du*cap;jv=dv*cap;ju=ju-(ju*prob).sum(-1,keepdim=True);jv=jv-(jv*prob).sum(-1,keepdim=True)
    eu=(prob*ju.square()).sum(-1);ev=(prob*jv.square()).sum(-1);cross=2*(prob*ju*jv).sum(-1);total=(prob*(ju+jv).square()).sum(-1);energy[hi,ci,:,sl]=torch.stack([eu,ev,cross,total]).cpu().numpy();replay=max(replay,float(((eu+ev+cross-total).abs()/total).max()))
    for ti,t in enumerate(cfg['amplitudes']):
     targetlp=(30*torch.tanh((raw+t*(du+dv))/30)).log_softmax(-1);p=targetlp.exp()
     for pi,dd in enumerate([torch.zeros_like(du),du,dv]):
      patchlp=(30*torch.tanh((raw+t*dd)/30)).log_softmax(-1);kl[hi,ci,ti,pi,sl]=(p*(targetlp-patchlp)).sum(-1).cpu().numpy()
  print('head',h,'complete',flush=True)
 assert np.isfinite(kl).all() and kl.min()>-1e-10 and replay<1e-10 and orth<1e-10
 old=old[:,:,[0,1,2]];error=float(np.max(np.abs(kl[:,:,0].mean(-1)-old.mean(-1))/old.mean(-1)));assert error<.002
 np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),kl=kl,energy=energy)
 analyze(cfg,kl,energy,dict(energy_identity_max_relative=replay,euclidean_cosine_max=orth,h366_mean_kl_replay_max_relative=error))
def analyze(cfg,kl,energy,audit):
 groups=[slice(0,1),slice(1,7),slice(7,13)];kv=kl.reshape(3,13,3,3,128,4);ev=energy.reshape(3,13,4,128,4)
 def stats(kk,ee):
  recovery=1-kk[:,:,:,1:]/kk[:,:,:,:1];local=1-ee[:,:,[1,0]]/ee[:,:,3,None];parts=ee[:,:,:3]/ee[:,:,3,None];return recovery,local,parts
 km=np.stack([kv[:,g].mean((1,4,5)) for g in groups],1);em=np.stack([ev[:,g].mean((1,3,4)) for g in groups],1);recovery,local,parts=stats(km,em)
 rng=np.random.default_rng(cfg['bootstrap_seed']);bs=[]
 for _ in range(cfg['bootstrap_samples']):
  ix=rng.integers(0,128,128);ek=[];kk=[]
  for g in groups:
   k=kv[:,g];e=ev[:,g];rr=rng.integers(0,k.shape[1],k.shape[1]);kk.append(k[:,rr][:,:,:,:,ix].mean((1,4,5)));ek.append(e[:,rr][:,:,:,ix].mean((1,3,4)))
  bs.append(stats(np.stack(kk,1),np.stack(ek,1))[2][:,:,2])
 bs=np.stack(bs);cross=parts[:,:,2];radius=float(np.quantile(np.max(np.abs(bs-cross),axis=(1,2)),.95));diff=np.max(np.abs(recovery-local[:,:,None]),axis=(0,1,3));gates={'small_amplitude_local':bool(diff[2]<.03),'full_amplitude_local':bool(diff[0]<.05),'head6_control_cancellation':bool((cross[2,1:]+radius<-.2).all())}
 j={'config':cfg,'families':['actual','moment','donor'],'finite_recovery':recovery.tolist(),'local_recovery':local.tolist(),'energy_parts_over_total':parts.tolist(),'cross_joint_ci95':np.stack([cross-radius,cross+radius],-1).tolist(),'max_recovery_difference_by_amplitude':diff.tolist(),'mean_kl':km.tolist(),'mean_energy':em.tolist(),'audit':audit,'gates':gates};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');print('FINISHED',gates,'maxdiff',diff,'cross',cross,flush=True)
if __name__=='__main__':main()

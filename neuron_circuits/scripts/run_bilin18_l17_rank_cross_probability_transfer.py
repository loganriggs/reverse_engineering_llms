"""H369 transfer probability metrics between contexts."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,B
STEM=ART/'bilin18_l17_rank_cross_probability_transfer'
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
 shared=((probabilities.reshape(128,4,-1).sum(0)[None]-probabilities.reshape(128,4,-1))/127).reshape(512,-1);assert float((shared.sum(-1)-1).abs().max())<1e-12 and shared.min()>=0
 cache=np.load(ART/'bilin18_l17_rank_prediction_confirmation_h0_rows.npz');indices=cache['indices'];assert indices.shape==(7,512) and np.array_equal(indices[0],np.arange(512)) and (indices[1:]//4!=np.arange(512)[None]//4).all() and (indices%4==np.arange(512)[None]%4).all()
 energy=np.empty((3,13,8,4,512));identity=0.;old=np.load(ART/'bilin18_l17_rank_cross_metric_factorial_rows.npz')['energy'][:,:,2]
 for hi,h in enumerate(cfg['heads']):
  cache=np.load(ART/f'bilin18_l17_rank_prediction_confirmation_h{h}_rows.npz');assert np.array_equal(cache['indices'],indices)
  for ci,response in enumerate(cache['responses']):
   target=F.rms_norm(state+torch.tensor(response,device='cuda',dtype=torch.float32),(1152,));delta=(target-baseline).double();u=(delta@V)@V.T;v=delta-u
   for off in range(0,512,16):
    sl=slice(off,off+16);du=u[sl]@W.T;dv=v[sl]@W.T
    for mi in range(8):
     prob=probabilities[torch.tensor(indices[mi,sl],device='cuda')] if mi<7 else shared[sl]
     ju=du-(du*prob).sum(-1,keepdim=True);jv=dv-(dv*prob).sum(-1,keepdim=True);eu=(prob*ju.square()).sum(-1);ev=(prob*jv.square()).sum(-1);cross=2*(prob*ju*jv).sum(-1);total=(prob*(ju+jv).square()).sum(-1);energy[hi,ci,mi,:,sl]=torch.stack([eu,ev,cross,total]).cpu().numpy();identity=max(identity,float(((eu+ev+cross-total).abs()/total).max()))
  print('head',h,'complete',flush=True)
 replay=float(np.max(np.abs(energy[:,:,0]-old))/np.max(np.abs(old)));assert replay<1e-10 and identity<1e-10 and np.isfinite(energy).all()
 np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),energy=energy,indices=indices)
 analyze(cfg,energy,{'h368_replay_relative_to_global_max':replay,'energy_identity_max_relative':identity,'shared_probability_sum_error':float((shared.sum(-1)-1).abs().max())})
def analyze(cfg,energy,audit):
 groups=[slice(0,1),slice(1,7),slice(7,13)];ev=energy.reshape(3,13,8,4,128,4)
 def stats(e,rr):
  e=np.stack([e[:,:,0],e[:,:,1+rr].mean(2),e[:,:,7]],2);z=e[:,:,:,2]/e[:,:,:,3];return z,z[:,:,1]-z[:,:,0],z[:,:,2]/z[:,:,0]
 em=np.stack([ev[:,g].mean((1,4,5)) for g in groups],1);z,contrast,retention=stats(em,np.arange(6));rng=np.random.default_rng(cfg['bootstrap_seed']);bz=[];bc=[];br=[]
 for _ in range(cfg['bootstrap_samples']):
  ix=rng.integers(0,128,128);rr=rng.integers(0,6,6);ee=[]
  for g in groups:
   e=ev[:,g];draw=rng.integers(0,e.shape[1],e.shape[1]);ee.append(e[:,draw][:,:,:,:,ix].mean((1,4,5)))
  zz,cc,rt=stats(np.stack(ee,1),rr);bz.append(zz);bc.append(cc);br.append(rt)
 def band(b,t):
  b=np.stack(b);radius=float(np.quantile(np.max(np.abs(b-t),axis=tuple(range(1,b.ndim))),.95));return np.stack([t-radius,t+radius],-1)
 zci=band(bz,z);cci=band(bc,contrast);rci=band(br,retention);gates={'shared_cancellation':bool((zci[:,:,2,1]<0).all()),'matched_attenuation':bool((cci[:,:,0]>.1).all()),'shared_majority':bool((rci[:,:,0]>.5).all())}
 j={'config':cfg,'families':['actual','moment','donor'],'metric_order':['own','confidence_matched','shared_loo_distribution'],'cross_over_total':z.tolist(),'cross_joint27_ci95':zci.tolist(),'matched_minus_own':contrast.tolist(),'matched_minus_own_joint9_ci95':cci.tolist(),'shared_over_own':retention.tolist(),'shared_over_own_joint9_ci95':rci.tolist(),'mean_energy_by_donor':em.tolist(),'audit':audit,'gates':gates};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');np.savez_compressed(STEM.with_name(STEM.name+'_bootstrap.npz'),cross=bz,contrast=bc,retention=br);print('FINISHED',gates,'cross',z,'contrast',contrast,'retention',retention,flush=True)
if __name__=='__main__':main()

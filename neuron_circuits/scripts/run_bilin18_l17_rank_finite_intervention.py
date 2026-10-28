"""H366 finite interventions after final RMS normalization."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,B
STEM=ART/'bilin18_l17_rank_finite_intervention'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,digest in cfg['source_sha256'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==digest
 assert hashlib.sha256((ROOT/cfg['corpus']).read_bytes()).hexdigest()==cfg['corpus_sha256']
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float()
 pre,_,_=collect_sources(model,np.load(ROOT/cfg['corpus'],mmap_mode='r'),cfg)
 L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]]
 x=F.rms_norm(pre,(1152,));state=pre+B(x,x,L,R,D)+model.blocks[17].mlp.Down_bias;baseline=F.rms_norm(state,(1152,))
 frozen=np.load(ART/'bilin18_l17_unembedding_rank_prediction_rows.npz');V=torch.tensor(frozen['eigenvectors'][:,:16],device='cuda',dtype=torch.float32);Q=torch.tensor(frozen['random_basis'][:,:2],device='cuda',dtype=torch.float32)
 assert float((V.T@V-torch.eye(16,device='cuda')).abs().max())<1e-5
 names=['baseline','keep2','remove2','keep16','random2','full'];kl=np.empty((3,13,6,512));matches=np.empty_like(kl)
 def lp(a):return (30*torch.tanh(model.lm_head(a).double()/30)).log_softmax(-1)
 for hi,h in enumerate(cfg['heads']):
  responses=np.load(ART/f'bilin18_l17_rank_prediction_confirmation_h{h}_rows.npz')['responses']
  for ci,response in enumerate(responses):
   target=F.rms_norm(state+torch.tensor(response,device='cuda',dtype=torch.float32),(1152,));delta=target-baseline;p2=(delta@V[:,:2])@V[:,:2].T;p16=(delta@V)@V.T;pq=(delta@Q)@Q.T
   variants=[baseline,baseline+p2,baseline+delta-p2,baseline+p16,baseline+pq,target]
   for off in range(0,512,16):
    sl=slice(off,off+16);t=lp(target[sl]);prob=t.exp()
    for ki,act in enumerate(variants):
     pred=lp(act[sl]);kl[hi,ci,ki,sl]=(prob*(t-pred)).sum(-1).cpu().numpy();matches[hi,ci,ki,sl]=(t.argmax(-1)==pred.argmax(-1)).cpu().numpy()
  print('head',h,'complete',flush=True)
 assert np.isfinite(kl).all() and kl.min()>-1e-10 and np.abs(kl[:,:,5]).max()<1e-12
 np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),kl=kl,top1_match=matches)
 rng=np.random.default_rng(cfg['bootstrap_seed']);values=kl.reshape(3,13,6,128,4);groups=[slice(0,1),slice(1,7),slice(7,13)]
 mean=np.stack([values[:,g].mean((1,3,4)) for g in groups],1);recovery=1-mean[:,:,1:5]/mean[:,:,:1];boot=[]
 for _ in range(cfg['bootstrap_samples']):
  ix=rng.integers(0,128,128);bm=[]
  for g in groups:
   v=values[:,g];draw=rng.integers(0,v.shape[1],v.shape[1]);bm.append(v[:,draw][:,:,:,ix].mean((1,3,4)))
  bm=np.stack(bm,1);boot.append(1-bm[:,:,1:5]/bm[:,:,:1])
 boot=np.stack(boot);radius=float(np.quantile(np.max(np.abs(boot-recovery),axis=(1,2,3)),.95));lower=recovery-radius;upper=recovery+radius;gap=recovery[:,:,0]-recovery[:,:,3];bg=boot[:,:,:,0]-boot[:,:,:,3];gr=float(np.quantile(np.max(np.abs(bg-gap),axis=(1,2)),.95))
 gates={'rank2_sufficient':bool((lower[:,:,0]>=.8).all()),'rank2_necessary':bool((upper[:,:,1]<=.5).all()),'rank16_sufficient':bool((lower[:,:,2]>=.8).all()),'rank2_orientation':bool((gap-gr>0).all())}
 result={'config':cfg,'patches':names,'families':['actual','moment','donor'],'mean_kl':mean.tolist(),'recovery':recovery.tolist(),'joint36_recovery_ci95':np.stack([lower,upper],-1).tolist(),'joint_recovery_radius':radius,'rank2_minus_random2':gap.tolist(),'rank2_minus_random2_joint_ci95':np.stack([gap-gr,gap+gr],-1).tolist(),'gates':gates,'top1_match':np.stack([matches[:,g].mean((1,3)) for g in groups],1).tolist()}
 np.savez_compressed(STEM.with_name(STEM.name+'_bootstrap.npz'),recovery=boot)
 STEM.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print('FINISHED',gates,'recovery',recovery.tolist(),flush=True)
if __name__=='__main__':main()

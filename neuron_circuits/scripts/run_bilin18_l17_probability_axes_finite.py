"""H373 compare probability-defined and weight-only output axes."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,B
STEM=ART/'bilin18_l17_probability_axes_finite'
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
 discovery=np.load(ART/'bilin18_l17_rank_covariance_mixture_frozen_probabilities.npz');p=torch.tensor(discovery['position_mean'].mean(0),device='cuda');means=torch.tensor(discovery['expected_unembedding'],device='cuda');K=W.T@(p[:,None]*W)-means.T@means/len(means);K=(K+K.T)/2;lam,V=torch.linalg.eigh(K);lam=lam.flip(0);V=V.flip(1);assert float(lam[-1])>-1e-9
 residual=float((K@V-V*lam).norm()/K.norm());orth=float((V.T@V-torch.eye(1152,device='cuda')).abs().max());assert residual<1e-10 and orth<1e-10
 np.savez_compressed(STEM.with_name(STEM.name+'_frozen_basis.npz'),metric=K.cpu().numpy(),eigenvalues=lam.cpu().numpy(),eigenvectors=V.cpu().numpy())
 weight=torch.tensor(np.load(ART/'bilin18_l17_unembedding_rank_prediction_rows.npz')['eigenvectors'],device='cuda');bases={'probability':V[:,:256],'weight':weight[:,:256]};mapped={k:v.T@W.T for k,v in bases.items()};names=['baseline']+[f'{k}{rank}' for k in bases for rank in cfg['ranks']];S=len(pre);kl=np.empty((3,13,9,S));local=np.empty_like(kl);null=0.
 for hi,h in enumerate(cfg['heads']):
  responses=np.load(ART/f'bilin18_l17_shared_geometry_precision_h{h}_responses.npz')['responses']
  for ci,response in enumerate(responses):
   target=F.rms_norm(state+torch.tensor(response,device='cuda'),(1152,));delta=(target-baseline).double();coeff={k:delta@v for k,v in bases.items()}
   for off in range(0,S,16):
    sl=slice(off,off+16);raw=baseline[sl].double()@W.T;dlog=delta[sl]@W.T;targetlp=(30*torch.tanh((raw+dlog)/30)).log_softmax(-1);tp=targetlp.exp();bp=(30*torch.tanh(raw/30)).softmax(-1);cap=1-torch.tanh(raw/30).square();variants=[torch.zeros_like(dlog)]+[coeff[k][sl,:rank]@mapped[k][:rank] for k in bases for rank in cfg['ranks']]
    for pi,change in enumerate(variants):
     pred=(30*torch.tanh((raw+change)/30)).log_softmax(-1);kl[hi,ci,pi,sl]=(tp*(targetlp-pred)).sum(-1).cpu().numpy();err=(dlog-change)*cap;err-= (err*bp).sum(-1,keepdim=True);local[hi,ci,pi,sl]=(bp*err.square()).sum(-1).cpu().numpy()
    null=max(null,float((tp*(targetlp-(30*torch.tanh((raw+dlog)/30)).log_softmax(-1))).sum(-1).abs().max()))
  print('head',h,'complete',flush=True)
 assert np.isfinite(kl).all() and kl.min()>-1e-10 and null<1e-12 and np.isfinite(local).all()
 np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),kl=kl,local_error=local)
 analyze(cfg,names,kl,local,{'eigen_relative_residual':residual,'orthogonality_max':orth,'minimum_eigenvalue':float(lam[-1]),'full_target_kl':null})
def analyze(cfg,names,kl,local,audit):
 groups=[slice(0,1),slice(1,7),slice(7,13)];kv=kl.reshape(3,13,9,cfg['rows'],4);lv=local.reshape(kv.shape);km=np.stack([kv[:,g].mean((1,3,4)) for g in groups],1);lm=np.stack([lv[:,g].mean((1,3,4)) for g in groups],1);recovery=1-km[:,:,1:]/km[:,:,:1];localrec=1-lm[:,:,1:]/lm[:,:,:1];rng=np.random.default_rng(cfg['bootstrap_seed']);bs=[]
 for _ in range(cfg['bootstrap_samples']):
  ix=rng.integers(0,cfg['rows'],cfg['rows']);kk=[]
  for g in groups:
   k=kv[:,g];rr=rng.integers(0,k.shape[1],k.shape[1]);kk.append(k[:,rr][:,:,:,ix].mean((1,3,4)))
  kk=np.stack(kk,1);bs.append(1-kk[:,:,1:]/kk[:,:,:1])
 bs=np.stack(bs);rad=float(np.quantile(np.max(np.abs(bs-recovery),axis=(1,2,3)),.95));ci=np.stack([recovery-rad,recovery+rad],-1);adv=recovery[:,:,1]-recovery[:,:,5];ba=bs[:,:,:,1]-bs[:,:,:,5];ar=float(np.quantile(np.max(np.abs(ba-adv),axis=(1,2)),.95));aci=np.stack([adv-ar,adv+ar],-1);gates={'probability64_sufficient':bool((ci[:,:,2,0]>=.8).all()),'probability256_sufficient':bool((ci[:,:,3,0]>=.8).all()),'probability16_advantage':bool((aci[:,:,0]>.1).all())}
 j={'config':cfg,'families':['actual','moment','donor'],'patches':names,'mean_kl':km.tolist(),'recovery':recovery.tolist(),'local_recovery':localrec.tolist(),'joint72_recovery_ci95':ci.tolist(),'probability16_advantage':adv.tolist(),'probability16_advantage_joint9_ci95':aci.tolist(),'audit':audit,'gates':gates};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');np.savez_compressed(STEM.with_name(STEM.name+'_bootstrap.npz'),recovery=bs);print('FINISHED',gates,'recovery',recovery,flush=True)
if __name__=='__main__':main()

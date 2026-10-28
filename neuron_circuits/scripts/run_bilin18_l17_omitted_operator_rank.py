"""H437 target-free source-restricted omitted operator ranks."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model
from run_bilin18_l17_fisher_ranked_factors import native_drms
STEM=ART/'bilin18_l17_omitted_operator_rank'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,h in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==h
 old=np.load(ART/'bilin18_l17_baseline_context_confirmation_rows.npz');model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();L,R,D=[m.weight.detach().double() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];W=model.lm_head.weight.double();Q=torch.linalg.qr(model.blocks[17].attn.c_proj.weight.reshape(1152,9,128)[:,6]).Q.double();pre=torch.tensor(old['pre'],device='cuda').double();st=torch.tensor(old['state'],device='cuda');base=F.rms_norm(st,(1152,));src=torch.tensor(old['sources'],device='cuda').double();cc=src@Q;sets=np.load(ART/'bilin18_l17_residual_factor_rescue_frozen_sets.npz');keep=torch.ones(4608,dtype=torch.bool,device='cuda');keep[torch.tensor(sets['sets128'][0],device='cuda')]=False;m=np.load(ART/'bilin18_l17_unmatched_moment_controls_moments.npz');mu=torch.tensor(m['mean'][1],device='cuda');cov=torch.tensor(m['covariance'][1],device='cuda');C=cov+mu[:,None]*mu[None,:];ev,U=torch.linalg.eigh(C);assert ev.min()>0;root=(U*ev.sqrt())@U.T;inverse=(U*ev.rsqrt())@U.T;white=cc@inverse;operators=[];grams=[];rawgrams=[]
 for si in range(32):
  p0=pre[si:si+1];x=F.rms_norm(p0,(1152,),eps=torch.finfo(torch.float32).eps);du=native_drms(p0,-Q.T);dp=(du@L.T)*(x@R.T)+(x@L.T)*(du@R.T);K=(dp[:,keep]@D[:,keep].T).T;Z=K@root;th=torch.tanh(base[si:si+1].double()@W.T/30);prob=(30*th).softmax(-1);dt=(native_drms(st[si:si+1].double(),Z.T)@W.T)*(1-th.square());dt-=(prob*dt).sum(-1,keepdim=True);G=.5*(dt*prob)@dt.T;operators.append(Z);grams.append(G);rawgrams.append(Z.T@Z);print('operator',si,'complete',flush=True)
 operators=torch.stack(operators);grams=torch.stack(grams);rawgrams=torch.stack(rawgrams);normgrams=grams/grams.diagonal(dim1=-2,dim2=-1).sum(-1)[:,None,None];rng=np.random.default_rng(cfg['random_seed']);random=torch.linalg.qr(torch.tensor(rng.standard_normal((128,128)),device='cuda')).Q;full=torch.tensor(old['full'],device='cuda').double();ret=torch.tensor(old['parts'][:,:,2],device='cuda').double();pred=np.empty((4,4,12,32,1152));spectra=[];bases=[];dense=[]
 for si in range(32):
  local=torch.linalg.eigh(grams[si]).eigenvectors.flip(-1);rawbasis=torch.linalg.eigh(rawgrams[si]).eigenvectors.flip(-1);shared=torch.linalg.eigh(normgrams.sum(0)-normgrams[si]).eigenvectors.flip(-1);dense.append(white@operators[si].T);spectra.append(torch.linalg.eigvalsh(grams[si]).flip(-1).cpu().numpy());basis=torch.stack([local,rawbasis,shared,random]);bases.append(basis.cpu().numpy())
  for method,B in enumerate(basis):
   for ri,rank in enumerate(cfg['ranks']):
    V=B[:,:rank];pred[method,ri,:,si]=((white@V)@(operators[si]@V).T).cpu().numpy()
 dense=torch.stack(dense,1);prior=np.load(ART/'bilin18_l17_omitted_local_prediction_rows.npz')['predictions'][0];replay=float(np.max(np.linalg.norm(dense.cpu().numpy()-prior,axis=-1)/np.maximum(np.linalg.norm(prior,axis=-1),1e-12)));errors=np.empty((4,4,12,32,2))
 for si in range(32):
  th=torch.tanh(base[si:si+1].double()@W.T/30);prob=(30*th).softmax(-1);cap=1-th.square();ss=st[si:si+1]
  def tangent(v):
   dt=(native_drms(ss.double(),v)@W.T)*cap;return dt-(prob*dt).sum(-1,keepdim=True)
  def logprob(v):return (30*torch.tanh(F.rms_norm(ss+v.float(),(1152,)).double()@W.T/30)).log_softmax(-1)
  tf=tangent(full[:,si]);lt=logprob(full[:,si]);pt=lt.exp()
  for method in range(4):
   for ri in range(4):
    prediction=ret[:,si]+torch.tensor(pred[method,ri,:,si],device='cuda');res=tf-tangent(prediction);lp=logprob(prediction);errors[method,ri,:,si]=torch.stack([.5*(prob*res.square()).sum(-1),(pt*(lt-lp)).sum(-1)],-1).cpu().numpy()
  print('evaluation',si,'complete',flush=True)
 assert np.isfinite(errors).all();np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),errors=errors,predictions=pred,spectra=np.stack(spectra),bases=np.stack(bases),operators=operators.cpu().numpy());j={'experiment':'H437','config':cfg,'dense_operator_replay_relative_max':replay,'dense_replay_gate':replay<1e-5,'source_second_moment_eigenvalues':[float(ev.min()),float(ev.max())],'unique_completed_documents':11968};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');print('FINISHED',replay,flush=True)
if __name__=='__main__':main()

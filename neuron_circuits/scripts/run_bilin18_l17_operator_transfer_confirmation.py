"""H438 frozen shared operator directions on independent contexts and sources."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources
from run_bilin18_l17_fisher_ranked_factors import native_drms
STEM=ART/'bilin18_l17_operator_transfer_confirmation'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,h in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==h
 assert hashlib.sha256((ROOT/cfg['corpus']).read_bytes()).hexdigest()==cfg['corpus_sha256'];model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();Lf,Rf,Df=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];L,R,D=Lf.double(),Rf.double(),Df.double();W=model.lm_head.weight.double();allpre,_,_=collect_sources(model,np.load(ROOT/cfg['corpus']),cfg);xx=F.rms_norm(allpre,(1152,));allstate=allpre+((xx@Lf.T)*(xx@Rf.T))@Df.T+model.blocks[17].mlp.Down_bias;rms=allstate.double().square().mean(-1).sqrt().cpu().numpy();low=np.argsort(rms,kind='stable')[:16];mid=np.array([i for i in np.argsort(abs(rms-np.median(rms)),kind='stable') if i not in low])[:16];sites=np.r_[low,mid];selection={'low':low.tolist(),'median':mid.tolist(),'output_rms':rms.tolist()};STEM.with_name(STEM.name+'_selection.json').write_text(json.dumps(selection,indent=2)+'\n');print('SELECTION FROZEN',sites,flush=True)
 pre=allpre[sites];st=allstate[sites];base=F.rms_norm(st,(1152,));_,dh,_=collect_sources(model,np.load(ROOT/cfg['donor_corpus']),cfg);Q=torch.linalg.qr(model.blocks[17].attn.c_proj.weight.reshape(1152,9,128)[:,6]).Q.double();mom=np.load(ART/'bilin18_l17_unmatched_moment_controls_moments.npz');mu=torch.tensor(mom['mean'][1],device='cuda');cov=torch.tensor(mom['covariance'][1],device='cuda');groot=torch.tensor(mom['root'][1],device='cuda');ev,U=torch.linalg.eigh(cov+mu[:,None]*mu[None,:]);assert ev.min()>0;root=(U*ev.sqrt())@U.T;inverse=(U*ev.rsqrt())@U.T;src=[];donor_ids=[]
 for seed in cfg['source_seeds']:
  di=int(np.random.default_rng(seed).integers(512));donor_ids.append(di);src.append(dh[di,6].double())
 for seed in cfg['source_seeds']:
  noise=torch.tensor(np.random.default_rng(seed+100000).standard_normal(128),device='cuda');src.append((mu+noise@groot.T)@Q.T)
 src=torch.stack(src);white=(src@Q)@inverse;shared=torch.tensor(np.load(ART/'bilin18_l17_omitted_operator_shared_frozen.npz')['basis'],device='cuda');random=torch.linalg.qr(torch.tensor(np.random.default_rng(cfg['random_seed']).standard_normal((128,128)),device='cuda')).Q;sets=np.load(ART/'bilin18_l17_residual_factor_rescue_frozen_sets.npz');ix=torch.tensor(sets['sets128'][0],device='cuda');keep=torch.ones(4608,dtype=torch.bool,device='cuda');keep[ix]=False;full=[];ret=[];xf=F.rms_norm(pre,(1152,));phi=(xf@Lf.T)*(xf@Rf.T)
 for h in src:
  xt=F.rms_norm(pre-h.float(),(1152,));phit=(xt@Lf.T)*(xt@Rf.T);full.append(phit@Df.T-phi@Df.T);ret.append((phit-phi)[:,ix]@Df[:,ix].T)
 full=torch.stack(full).double();ret=torch.stack(ret).double();errors=np.empty((7,24,32,2));effects=np.empty((24,32,2));predictions=np.empty((6,24,32,1152));replay=0.;spectra=[]
 for si in range(32):
  pp=pre[si:si+1].double();x=F.rms_norm(pp,(1152,),eps=torch.finfo(torch.float32).eps);lx=x@L.T;rx=x@R.T;du=native_drms(pp,-Q.T);dp=(du@L.T)*rx+lx*(du@R.T);K=(dp[:,keep]@D[:,keep].T).T;Z=K@root;ss=st[si:si+1];th=torch.tanh(base[si:si+1].double()@W.T/30);p=(30*th).softmax(-1);lp=(30*th).log_softmax(-1);cap=1-th.square()
  def tangent(v):
   dt=(native_drms(ss.double(),v)@W.T)*cap;return dt-(p*dt).sum(-1,keepdim=True)
  def logprob(v):return (30*torch.tanh(F.rms_norm(ss+v.float(),(1152,)).double()@W.T/30)).log_softmax(-1)
  dt=tangent(Z.T);G=.5*(dt*p)@dt.T;evals,vec=torch.linalg.eigh(G);local=vec.flip(-1);spectra.append(evals.flip(-1).cpu().numpy());dense=white@Z.T;dv=native_drms(pp,-src);dp=(dv@L.T)*rx+lx*(dv@R.T);direct=dp[:,keep]@D[:,keep].T;replay=max(replay,float(((dense-direct).norm(dim=-1)/direct.norm(dim=-1).clamp_min(1e-12)).max()));guesses=[]
  for B,rank in [(local,8),(local,16),(shared,8),(shared,16),(None,128),(random,16)]:
   guesses.append(dense if B is None else (white@B[:,:rank])@(Z@B[:,:rank]).T)
  tf=tangent(full[:,si]);lt=logprob(full[:,si]);pt=lt.exp();effects[:,si]=torch.stack([.5*(p*tf.square()).sum(-1),(pt*(lt-lp)).sum(-1)],-1).cpu().numpy()
  for method,g in enumerate(guesses+[torch.zeros_like(dense)]):
   if method<6:predictions[method,:,si]=g.cpu().numpy()
   v=ret[:,si]+g;res=tf-tangent(v);lg=logprob(v);errors[method,:,si]=torch.stack([.5*(p*res.square()).sum(-1),(pt*(lt-lg)).sum(-1)],-1).cpu().numpy()
  np.savez_compressed(STEM.with_name(STEM.name+'_partial_rows.npz'),errors=errors[:,:,:si+1],effects=effects[:,:si+1]);print('site',si,'complete',flush=True)
 assert np.isfinite(errors).all() and np.isfinite(effects).all();np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),errors=errors,effects=effects,predictions=predictions,full=full.cpu().numpy(),retained=ret.cpu().numpy(),sources=src.cpu().numpy(),pre=pre.cpu().numpy(),state=st.cpu().numpy(),spectra=np.stack(spectra));j={'experiment':'H438','config':cfg,'selection':selection,'donor_ids':donor_ids,'dense_operator_replay_relative_max':replay,'dense_replay_gate':replay<1e-5,'document_audit':json.loads((ART/'bilin18_operator_transfer_confirmation_document_audit.json').read_text()),'unique_completed_documents':12480};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');print('FINISHED',replay,flush=True)
if __name__=='__main__':main()

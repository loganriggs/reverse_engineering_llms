"""H459 cross-head scope of frozen head6 source operation."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources
from run_bilin18_l17_fisher_ranked_factors import native_drms
STEM=ART/'bilin18_l17_frozen_operation_allheads'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,h in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==h
 oldcfg=cfg;assert hashlib.sha256((ROOT/cfg['corpus']).read_bytes()).hexdigest()==cfg['corpus_sha256'];sites=np.array(json.loads((ART/'bilin18_l17_unprojected_finite_confirmation_selection.json').read_text())['sites']);STEM.with_name(STEM.name+'_selection.json').write_text(json.dumps({'sites':sites.tolist(),'scope':'Reuse H458128 receiving contexts with fresh cross-head donor draws'},indent=2)+'\n')
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();Lf,Rf,Df=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];L,R,D=Lf.double(),Rf.double(),Df.double();W=model.lm_head.weight.double();allpre,_,_=collect_sources(model,np.load(ROOT/oldcfg['corpus']),oldcfg);pre=allpre[sites];xf=F.rms_norm(pre,(1152,));phi=(xf@Lf.T)*(xf@Rf.T);st=pre+phi@Df.T+model.blocks[17].mlp.Down_bias;base=F.rms_norm(st,(1152,));_,dh,_=collect_sources(model,np.load(ROOT/oldcfg['donor_corpus']),oldcfg);Q=torch.linalg.qr(model.blocks[17].attn.c_proj.weight.reshape(1152,9,128)[:,6]).Q.double();m=np.load(ART/'bilin18_l17_unmatched_moment_controls_moments.npz');mu=torch.tensor(m['mean'][1],device='cuda');cov=torch.tensor(m['covariance'][1],device='cuda');gr=torch.tensor(m['root'][1],device='cuda');ev,U=torch.linalg.eigh(cov+mu[:,None]*mu[None,:]);inv=(U*ev.rsqrt())@U.T;V=torch.tensor(np.load(ART/'bilin18_l17_omitted_operator_shared_frozen.npz')['basis'][:,:16],device='cuda');H=torch.tensor(np.load(ART/'bilin18_l17_compiled_omitted_maps_weight_compilation.npz')['source_directions'],device='cuda');T=torch.tensor(np.load(ART/'bilin18_l17_compiled_omitted_maps_weight_compilation.npz')['maps'],device='cuda');sources=[];donors=[]
 for seed in cfg['source_seeds']:donors.append(int(np.random.default_rng(seed).integers(512)))
 for head in range(9):
  for di in donors:sources.append(dh[di,head].double())
 sources=torch.stack(sources);coeff=sources@Q@inv@V;ix=torch.tensor(np.load(ART/'bilin18_l17_residual_factor_rescue_frozen_sets.npz')['sets128'][0],device='cuda');s=(pre.double().square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();x=pre.double()/s;phid=(x@L.T)*(x@R.T);mo=phid@D.T-phid[:,ix]@D[:,ix].T;A=torch.einsum('roi,ni->nro',T,x);linear=-torch.einsum('ar,nro->ano',coeff,A)/s[None];rad=2*torch.einsum('ar,nr->an',coeff,x@H/1152)[:,:,None]*mo[None]/s[None];full=[];ret=[]
 z=np.load(ART/'bilin18_l17_joint_input_regression_rank256_compiled.npz');enc=torch.tensor(z['encoder'],device='cuda');tc=torch.tensor(z['maps'],device='cuda');compact=-torch.einsum('ar,roj,nj->ano',coeff,tc,x@enc.T)/s[None];targets=torch.stack([linear,compact]);qz=np.load(ART/'bilin18_l17_compiled_finite_source_law_quadratic_compiled.npz');pairwrites=torch.tensor(qz['writes'],device='cuda');gram=torch.tensor(qz['source_gram'],device='cuda');pairs=qz['pairs'];products=torch.stack([coeff[:,i]*coeff[:,j] for i,j in pairs],1);quad=products@pairwrites;hhat=coeff@H.T;hsq=((coeff@gram)*coeff).sum(-1);amp=torch.tensor(cfg['amplitudes'],device='cuda');allfull=[];allret=[]
 for alpha in cfg['amplitudes']:
  fs=[];rs=[]
  for h in sources:
   xt=F.rms_norm(pre-alpha*h.float(),(1152,));phit=(xt@Lf.T)*(xt@Rf.T);fs.append(phit@Df.T-phi@Df.T);rs.append((phit-phi)[:,ix]@Df[:,ix].T)
  allfull.append(torch.stack(fs));allret.append(torch.stack(rs))
 allfull=torch.stack(allfull).double();allret=torch.stack(allret).double();errors=np.empty((2,3,108,128,2));effects=np.empty((2,108,128,2));raw_errors=np.empty((2,3,108,128));raw_effects=np.empty((2,108,128));rational_audit=0.;native_audit=0.
 for si in range(128):
  ss=st[si:si+1];th=torch.tanh(base[si:si+1].double()@W.T/30);p=(30*th).softmax(-1);cap=1-th.square();lp0=(30*th).log_softmax(-1)
  def tangent(v):
   t=(native_drms(ss.double(),v)@W.T)*cap;return t-(p*t).sum(-1,keepdim=True)
  def lp(v):return (30*torch.tanh(F.rms_norm(ss+v.float(),(1152,)).double()@W.T/30)).log_softmax(-1)
  aa=amp[:,None,None];cross=(coeff@(pre[si].double()@H/1152))[:,None];den=s[si].square()-2*aa*cross[None]+aa.square()*hsq[None,:,None];radial=(s[si].square()/den-1)*mo[si][None,None,:];other=aa[None]*s[si].square()*targets[:,:,si][:,None]/den[None]+aa[None].square()*quad[None,None]/den[None]
  if si in [0,32,64,96]:
   pp=pre[si].double()[None,None]-aa*hhat[None];qq=pp/(pp.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();pc=(qq@L.T)*(qq@R.T);od=pc@D.T-pc[:,:,ix]@D[:,ix].T-mo[si];exact=other[0]+radial;rational_audit=max(rational_audit,float((exact-od).norm()/od.norm()));qf=F.rms_norm(pre[si][None,None]-aa.float()*hhat.float()[None],(1152,));pf=(qf@Lf.T)*(qf@Rf.T);native=(pf@Df.T-pf[:,:,ix]@Df[:,ix].T)-(phi[si]@Df.T-phi[si,ix]@Df[:,ix].T);native_audit=max(native_audit,float((exact-native.double()).norm()/exact.norm()))
  f=allfull[:,:,si].reshape(-1,1152);tf=tangent(f);lt=lp(f);pt=lt.exp();effects[:,:,si]=torch.stack([.5*(p*tf.square()).sum(-1),(pt*(lt-lp0)).sum(-1)],-1).reshape(2,108,2).cpu().numpy();raw_effects[:,:,si]=f.square().sum(-1).reshape(2,108).cpu().numpy()
  for mi in range(3):
   v=allret[:,:,si] if mi==2 else allret[:,:,si]+radial+other[mi];v=v.reshape(-1,1152);res=tf-tangent(v);lg=lp(v);errors[:,mi,:,si]=torch.stack([.5*(p*res.square()).sum(-1),(pt*(lt-lg)).sum(-1)],-1).reshape(2,108,2).cpu().numpy();raw_errors[:,mi,:,si]=(f-v).square().sum(-1).reshape(2,108).cpu().numpy()
  np.savez_compressed(STEM.with_name(STEM.name+'_partial_rows.npz'),errors=errors[:,:,:,:si+1],effects=effects[:,:,:si+1]);print('site',si,'complete',flush=True)
 assert all(np.isfinite(v).all() for v in [errors,effects,raw_errors,raw_effects]);np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),errors=errors,effects=effects,raw_errors=raw_errors,raw_effects=raw_effects,source_reconstruction_error=(sources-hhat).square().sum(-1).reshape(9,12).cpu().numpy(),source_energy=sources.square().sum(-1).reshape(9,12).cpu().numpy(),sources=sources.cpu().numpy(),pre=pre.cpu().numpy(),state=st.cpu().numpy());j={'experiment':'H459','config':cfg,'methods':['dense_finite','joint256_finite','retained_only'],'sites':sites.tolist(),'donor_ids':donors,'rational_relative_max':rational_audit,'native_projected_relative_max':native_audit,'numerical_gates':{'rational':rational_audit<1e-10,'native_projected':native_audit<1e-4},'unique_completed_documents':12992};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');print('FINISHED',j['numerical_gates'],flush=True)
if __name__=='__main__':main()

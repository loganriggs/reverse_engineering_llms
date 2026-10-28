"""H451 shared and zero tail mean approximation."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model
from run_bilin18_l17_fisher_ranked_factors import native_drms
STEM=ART/'bilin18_l17_tail_mean_approximation'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,h in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==h
 old=np.load(ART/'bilin18_l17_scalar_tail_confirmation_rows.npz');projection=np.load(ART/'bilin18_l17_metric_compiled_selection_frozen_projection.npz');selection=json.loads((ART/'bilin18_l17_metric_compiled_selection_selection.json').read_text());enc=torch.tensor(projection['encoders'][selection['selected']],device='cuda');dec=torch.tensor(projection['decoders'][selection['selected']],device='cuda');learned=torch.linalg.qr(dec).Q;random=torch.linalg.qr(torch.tensor(np.random.default_rng(cfg['random_seeds'][0]).standard_normal((1152,64)),device='cuda')).Q;model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();L,R,D=[m.weight.detach().double() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];W=model.lm_head.weight.double();pre=torch.tensor(old['pre'],device='cuda').double();st=torch.tensor(old['state'],device='cuda');base=F.rms_norm(st,(1152,));s=(pre.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();x=pre/s;ix=torch.tensor(np.load(ART/'bilin18_l17_residual_factor_rescue_frozen_sets.npz')['sets128'][0],device='cuda');phi=(x@L.T)*(x@R.T);mo=phi@D.T-phi[:,ix]@D[:,ix].T;Q=torch.linalg.qr(model.blocks[17].attn.c_proj.weight.reshape(1152,9,128)[:,6]).Q.double();m=np.load(ART/'bilin18_l17_unmatched_moment_controls_moments.npz');mu=torch.tensor(m['mean'][1],device='cuda');cov=torch.tensor(m['covariance'][1],device='cuda');ev,U=torch.linalg.eigh(cov+mu[:,None]*mu[None,:]);inv=(U*ev.rsqrt())@U.T;V=torch.tensor(np.load(ART/'bilin18_l17_omitted_operator_shared_frozen.npz')['basis'][:,:16],device='cuda');src=torch.tensor(old['sources'],device='cuda');coeff=src@Q@inv@V;H=torch.tensor(np.load(ART/'bilin18_l17_compiled_omitted_maps_weight_compilation.npz')['source_directions'],device='cuda');sc=2*torch.einsum('ar,nr->an',coeff,x@H/1152)/s.T;rad=sc[:,:,None]*mo[None];maps=torch.tensor(np.load(ART/'bilin18_l17_compiled_omitted_maps_weight_compilation.npz')['maps'],device='cuda');compiled=-torch.einsum('ar,roi,ni->ano',coeff,maps,x)/s[None]+rad;linear=compiled-rad;full=torch.tensor(old['full'],device='cuda').double();ret=torch.tensor(old['retained'],device='cuda').double();errors=np.empty((6,24,32,2));normratios=np.empty((6,24,32));local_linear_error=np.empty((6,24,32));audit=0.;condition=[];predictions=np.empty((6,24,32,1152))
 bases=[learned]+[torch.linalg.qr(torch.tensor(np.random.default_rng(seed).standard_normal((1152,64)),device='cuda')).Q for seed in cfg['random_seeds']]
 calstate=torch.tensor(np.load(ART/'bilin18_l17_context_projection_transfer_rows.npz')['state'],device='cuda');means=[]
 for cs in calstate:
  ct=torch.tanh(F.rms_norm(cs[None],(1152,)).double()@W.T/30);cp=(30*ct).softmax(-1)[0];cc=(1-ct.square())[0];ids=cp.argsort(descending=True)[128:];ww=cp[ids]/cp[ids].sum();means.append((ww*cc[ids])@W[ids])
 sharedmean=torch.stack(means).mean(0);constant=float(np.load(ART/'bilin18_l17_scalar_tail_confirmation_calibration.npz')['constant']);np.savez_compressed(STEM.with_name(STEM.name+'_calibration.npz'),means=torch.stack(means).cpu().numpy(),sharedmean=sharedmean.cpu().numpy(),constant=constant);receiving_means=[];mean_audit=0.
 labels=['full_condition','constant_exact_tail_mean','constant_shared_tail_mean','constant_zero_tail_mean','constant_no_between','top128_condition'];errors=np.empty((6,9,24,128,2));norms=np.empty((6,9,24,128));split=np.empty((6,9,24,128,2));mass=[];audit=0.;decomp=0.;lambdas=np.empty((6,9,128))
 for si in range(128):
  ss=st[si:si+1];th=torch.tanh(base[si:si+1].double()@W.T/30);p=(30*th).softmax(-1);cap=1-th.square();order=p[0].argsort(descending=True);head=order[:128];tail=order[128:];mh=p[0,head].sum();mt=p[0,tail].sum();mass.append(float(mh));wh=p[0,head]/mh;wt=p[0,tail]/mt;perm=torch.tensor(np.random.default_rng(cfg['control_seed']+si).permutation(len(tail)),device='cuda');ws=wt[perm]
  def tangent_raw(v):return (native_drms(ss.double(),v)@W.T)*cap
  def tangent(v):
   t=tangent_raw(v);return t-(p*t).sum(-1,keepdim=True)
  def lp(v):return (30*torch.tanh(F.rms_norm(ss+v.float(),(1152,)).double()@W.T/30)).log_softmax(-1)
  tl=tangent_raw(linear[:,si]);tf=tangent(full[:,si]);lt=lp(full[:,si]);pt=lt.exp()
  for bi,b in enumerate(bases):
   tb=tangent_raw(b.T)
   def stats(ids,w):
    bb=tb[:,ids];ll=tl[:,ids];bm=(bb*w).sum(-1);lm=(ll*w).sum(-1);bc=bb-bm[:,None];lc=ll-lm[:,None];return .5*(bc*w)@bc.T,.5*(bc*w)@lc.T,bm,lm
   gh,rh,bh,lh=stats(head,wh);gt,rt,bt,ltail=stats(tail,wt);gs,rs,bs,ls=stats(tail,ws)
   def between(bm,lm):
    db=bh-bm;dl=lh-lm;return .5*mh*mt*db[:,None]*db[None,:],.5*mh*mt*db[:,None]*dl[None,:]
   gb,rb=between(bt,ltail);gbs,rbs=between(bs,ls);gf=mh*gh+mt*gt+gb;rf=mh*rh+mt*rt+rb;bc=tb-(tb*p).sum(-1,keepdim=True);lc=tl-(tl*p).sum(-1,keepdim=True);gd=.5*(bc*p)@bc.T;rd=.5*(bc*p)@lc.T;decomp=max(decomp,float((gf-gd).norm()/gd.norm()),float((rf-rd).norm()/rd.norm()))
   dzb=native_drms(ss.double(),b.T);dzl=native_drms(ss.double(),linear[:,si]);mg=(wt*cap[0,tail])@W[tail]
   if bi==0:receiving_means.append(mg.cpu().numpy())
   mean_audit=max(mean_audit,float((dzb@mg-bt).norm()/bt.norm()),float((dzl@mg-ltail).norm()/ltail.norm()));gbs,rbs=between(dzb@sharedmean,dzl@sharedmean);gbz,rbz=between(torch.zeros_like(bt),torch.zeros_like(ltail));gi=constant/1152*(dzb@dzb.T);ri=constant/1152*(dzb@dzl.T);gg=mh*gh+mt*gi;rr=mh*rh+mt*ri;systems=[(gf,rf),(gg+gb,rr+rb),(gg+gbs,rr+rbs),(gg+gbz,rr+rbz),(gg,rr),(gh,rh)]
   for ci,(g,rhs) in enumerate(systems):
    ev=torch.linalg.eigvalsh(g);lam=cfg['ridge']*g.trace()/64
    lam=torch.maximum(lam,(ev[-1]-cfg['condition_cap']*ev[0])/(cfg['condition_cap']-1))
    reg=g+lam*torch.eye(64,device='cuda');c=torch.linalg.solve(reg,rhs);audit=max(audit,float((reg@c-rhs).norm()/rhs.norm()));cor=c.T@b.T;v=ret[:,si]+rad[:,si]+cor;res=tf-tangent(v);lg=lp(v);errors[ci,bi,:,si]=torch.stack([.5*(p*res.square()).sum(-1),(pt*(lt-lg)).sum(-1)],-1).cpu().numpy();norms[ci,bi,:,si]=(cor.norm(dim=-1)/linear[:,si].norm(dim=-1)).cpu().numpy();en=.5*p*res.square();inside=en[:,head].sum(-1);split[ci,bi,:,si]=torch.stack([inside,en.sum(-1)-inside],-1).cpu().numpy();lambdas[ci,bi,si]=float(lam/(g.trace()/64))
  np.savez_compressed(STEM.with_name(STEM.name+'_partial_rows.npz'),errors=errors[:,:,:,:si+1],norms=norms[:,:,:,:si+1]);print('site',si,'complete',flush=True)
 replay=max(float(abs(errors[0]-old['errors'][0]).max()),float(abs(errors[1]-old['errors'][3]).max()),float(abs(errors[4]-old['errors'][5]).max()),float(abs(errors[5]-old['errors'][1]).max()));assert all(np.isfinite(v).all() for v in [errors,norms,split,lambdas]);np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),errors=errors,norms=norms,split_error=split,lambdas=lambdas,masses=mass,receiving_means=receiving_means);j={'experiment':'H451','config':cfg,'conditions':labels,'mean_transport_relative_max':mean_audit,'solve_relative_max':audit,'decomposition_relative_max':decomp,'replay_max_abs':replay,'numerical_gates':{'mean_transport':mean_audit<1e-10,'solve':audit<1e-8,'decomposition':decomp<1e-10,'replay':replay<1e-8},'unique_completed_documents':12480};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');print('FINISHED',j['numerical_gates'],flush=True)
if __name__=='__main__':main()

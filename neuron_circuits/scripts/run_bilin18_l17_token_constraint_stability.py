"""H446 frozen ridge and spectral audit of token constraint stability."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model
from run_bilin18_l17_fisher_ranked_factors import native_drms
STEM=ART/'bilin18_l17_token_constraint_stability'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,h in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==h
 old=np.load(ART/'bilin18_l17_context_projection_transfer_rows.npz');projection=np.load(ART/'bilin18_l17_metric_compiled_selection_frozen_projection.npz');selection=json.loads((ART/'bilin18_l17_metric_compiled_selection_selection.json').read_text());enc=torch.tensor(projection['encoders'][selection['selected']],device='cuda');dec=torch.tensor(projection['decoders'][selection['selected']],device='cuda');learned=torch.linalg.qr(dec).Q;random=torch.linalg.qr(torch.tensor(np.random.default_rng(cfg['random_seeds'][0]).standard_normal((1152,64)),device='cuda')).Q;model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();L,R,D=[m.weight.detach().double() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];W=model.lm_head.weight.double();pre=torch.tensor(old['pre'],device='cuda').double();st=torch.tensor(old['state'],device='cuda');base=F.rms_norm(st,(1152,));s=(pre.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();x=pre/s;ix=torch.tensor(np.load(ART/'bilin18_l17_residual_factor_rescue_frozen_sets.npz')['sets128'][0],device='cuda');phi=(x@L.T)*(x@R.T);mo=phi@D.T-phi[:,ix]@D[:,ix].T;Q=torch.linalg.qr(model.blocks[17].attn.c_proj.weight.reshape(1152,9,128)[:,6]).Q.double();m=np.load(ART/'bilin18_l17_unmatched_moment_controls_moments.npz');mu=torch.tensor(m['mean'][1],device='cuda');cov=torch.tensor(m['covariance'][1],device='cuda');ev,U=torch.linalg.eigh(cov+mu[:,None]*mu[None,:]);inv=(U*ev.rsqrt())@U.T;V=torch.tensor(np.load(ART/'bilin18_l17_omitted_operator_shared_frozen.npz')['basis'][:,:16],device='cuda');src=torch.tensor(old['sources'],device='cuda');coeff=src@Q@inv@V;H=torch.tensor(np.load(ART/'bilin18_l17_compiled_omitted_maps_weight_compilation.npz')['source_directions'],device='cuda');sc=2*torch.einsum('ar,nr->an',coeff,x@H/1152)/s.T;rad=sc[:,:,None]*mo[None];maps=torch.tensor(np.load(ART/'bilin18_l17_compiled_omitted_maps_weight_compilation.npz')['maps'],device='cuda');compiled=-torch.einsum('ar,roi,ni->ano',coeff,maps,x)/s[None]+rad;linear=compiled-rad;full=torch.tensor(old['full'],device='cuda').double();ret=torch.tensor(old['retained'],device='cuda').double();errors=np.empty((6,24,32,2));normratios=np.empty((6,24,32));local_linear_error=np.empty((6,24,32));audit=0.;condition=[];predictions=np.empty((6,24,32,1152))
 bases=[learned]+[torch.linalg.qr(torch.tensor(np.random.default_rng(seed).standard_normal((1152,64)),device='cuda')).Q for seed in cfg['random_seeds']]
 conditions=[(0,1e-6)]+[(k,r) for k in cfg['budgets'] for r in cfg['ridges']]
 errors=np.empty((10,5,24,128,2)); norms=np.empty((10,5,24,128)); split_error=np.empty((10,5,24,128,2)); spectra=np.empty((4,5,128,64)); effective=np.empty((10,5,128)); solve=0.
 for si in range(128):
  ss=st[si:si+1];th=torch.tanh(base[si:si+1].double()@W.T/30);p=(30*th).softmax(-1);cap=1-th.square();top=p[0].argsort(descending=True)
  def tangent_raw(v):return (native_drms(ss.double(),v)@W.T)*cap
  def tangent(v):
   t=tangent_raw(v);return t-(p*t).sum(-1,keepdim=True)
  def lp(v):return (30*torch.tanh(F.rms_norm(ss+v.float(),(1152,)).double()@W.T/30)).log_softmax(-1)
  tl=tangent_raw(linear[:,si]);tf=tangent(full[:,si]);lt=lp(full[:,si]);pt=lt.exp()
  for bi,b in enumerate(bases):
   tb=tangent_raw(b.T)
   for ki,k in enumerate([0]+cfg['budgets']):
    ids=torch.arange(W.shape[0],device='cuda') if k==0 else top[:k];weights=p[0,ids];weights=weights/weights.sum();bb=tb[:,ids];ll=tl[:,ids];bb=bb-(bb*weights).sum(-1,keepdim=True);ll=ll-(ll*weights).sum(-1,keepdim=True);g=.5*(bb*weights)@bb.T;rhs=.5*(bb*weights)@ll.T;ev=torch.linalg.eigvalsh(g);spectra[ki,bi,si]=ev.cpu().numpy()
    for ridge in ([1e-6] if k==0 else cfg['ridges']):
     ci=conditions.index((k,ridge));lam=ridge*g.trace()/64;reg=g+lam*torch.eye(64,device='cuda');c=torch.linalg.solve(reg,rhs);solve=max(solve,float((reg@c-rhs).norm()/rhs.norm()));correction=c.T@b.T;v=ret[:,si]+correction+rad[:,si];res=tf-tangent(v);lg=lp(v);errors[ci,bi,:,si]=torch.stack([.5*(p*res.square()).sum(-1),(pt*(lt-lg)).sum(-1)],-1).cpu().numpy();norms[ci,bi,:,si]=(correction.norm(dim=-1)/linear[:,si].norm(dim=-1)).cpu().numpy();effective[ci,bi,si]=float((ev.clamp_min(0)/(ev.clamp_min(0)+lam)).sum());energy=.5*p*res.square();inside=energy[:,ids].sum(-1);split_error[ci,bi,:,si]=torch.stack([inside,energy.sum(-1)-inside],-1).cpu().numpy()
  np.savez_compressed(STEM.with_name(STEM.name+'_partial_rows.npz'),errors=errors[:,:,:,:si+1],norms=norms[:,:,:,:si+1],spectra=spectra[:,:,:si+1],effective=effective[:,:,:si+1],split_error=split_error[:,:,:,:si+1]);print('site',si,'complete',flush=True)
 prior=np.load(ART/'bilin18_l17_token_constraint_budget_rows.npz')['errors'];replay=max(float(abs(errors[i]-prior[q]).max()) for i,q in [(0,0),(1,1),(4,2),(7,3)])
 assert all(np.isfinite(v).all() for v in [errors,norms,spectra,effective,split_error]);np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),errors=errors,norms=norms,spectra=spectra,effective=effective,split_error=split_error);j={'experiment':'H446','config':cfg,'conditions':conditions,'full_replay_max_abs':replay,'solve_relative_max':solve,'numerical_gates':{'replay':replay<1e-8,'solve':solve<1e-8},'unique_completed_documents':12480};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');print('FINISHED',j['numerical_gates'],flush=True)
if __name__=='__main__':main()

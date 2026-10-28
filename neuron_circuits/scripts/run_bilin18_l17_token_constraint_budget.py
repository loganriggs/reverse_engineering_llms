"""H443 fixed output spans with baseline-context projection coefficients."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model
from run_bilin18_l17_fisher_ranked_factors import native_drms
STEM=ART/'bilin18_l17_token_constraint_budget'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,h in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==h
 old=np.load(ART/'bilin18_l17_context_projection_transfer_rows.npz');projection=np.load(ART/'bilin18_l17_metric_compiled_selection_frozen_projection.npz');selection=json.loads((ART/'bilin18_l17_metric_compiled_selection_selection.json').read_text());enc=torch.tensor(projection['encoders'][selection['selected']],device='cuda');dec=torch.tensor(projection['decoders'][selection['selected']],device='cuda');learned=torch.linalg.qr(dec).Q;random=torch.linalg.qr(torch.tensor(np.random.default_rng(cfg['random_seeds'][0]).standard_normal((1152,64)),device='cuda')).Q;model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();L,R,D=[m.weight.detach().double() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];W=model.lm_head.weight.double();pre=torch.tensor(old['pre'],device='cuda').double();st=torch.tensor(old['state'],device='cuda');base=F.rms_norm(st,(1152,));s=(pre.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();x=pre/s;ix=torch.tensor(np.load(ART/'bilin18_l17_residual_factor_rescue_frozen_sets.npz')['sets128'][0],device='cuda');phi=(x@L.T)*(x@R.T);mo=phi@D.T-phi[:,ix]@D[:,ix].T;Q=torch.linalg.qr(model.blocks[17].attn.c_proj.weight.reshape(1152,9,128)[:,6]).Q.double();m=np.load(ART/'bilin18_l17_unmatched_moment_controls_moments.npz');mu=torch.tensor(m['mean'][1],device='cuda');cov=torch.tensor(m['covariance'][1],device='cuda');ev,U=torch.linalg.eigh(cov+mu[:,None]*mu[None,:]);inv=(U*ev.rsqrt())@U.T;V=torch.tensor(np.load(ART/'bilin18_l17_omitted_operator_shared_frozen.npz')['basis'][:,:16],device='cuda');src=torch.tensor(old['sources'],device='cuda');coeff=src@Q@inv@V;H=torch.tensor(np.load(ART/'bilin18_l17_compiled_omitted_maps_weight_compilation.npz')['source_directions'],device='cuda');sc=2*torch.einsum('ar,nr->an',coeff,x@H/1152)/s.T;rad=sc[:,:,None]*mo[None];maps=torch.tensor(np.load(ART/'bilin18_l17_compiled_omitted_maps_weight_compilation.npz')['maps'],device='cuda');compiled=-torch.einsum('ar,roi,ni->ano',coeff,maps,x)/s[None]+rad;linear=compiled-rad;full=torch.tensor(old['full'],device='cuda').double();ret=torch.tensor(old['retained'],device='cuda').double();errors=np.empty((6,24,32,2));normratios=np.empty((6,24,32));local_linear_error=np.empty((6,24,32));audit=0.;condition=[];predictions=np.empty((6,24,32,1152))
 bases=[learned]+[torch.linalg.qr(torch.tensor(np.random.default_rng(seed).standard_normal((1152,64)),device='cuda')).Q for seed in cfg['random_seeds']];labels=['full','top32','top64','top128','top256','top512','shuffle_weights128','random_identities128'];errors=np.empty((8,5,24,128,2));masses=[];solve=0.
 for si in range(128):
  ss=st[si:si+1];th=torch.tanh(base[si:si+1].double()@W.T/30);p=(30*th).softmax(-1);cap=1-th.square()
  def tangent_raw(v):return (native_drms(ss.double(),v)@W.T)*cap
  def tangent(v):
   t=tangent_raw(v);return t-(p*t).sum(-1,keepdim=True)
  def lp(v):return (30*torch.tanh(F.rms_norm(ss+v.float(),(1152,)).double()@W.T/30)).log_softmax(-1)
  top=p[0].argsort(descending=True);masses.append([float(p[0,top[:k]].sum()) for k in cfg['budgets']]);tl=tangent_raw(linear[:,si]);tf=tangent(full[:,si]);lt=lp(full[:,si]);pt=lt.exp();rng=np.random.default_rng(cfg['control_seed']+si);permutation=torch.tensor(rng.permutation(128),device='cuda');randomids=torch.tensor(rng.choice(W.shape[0],128,replace=False),device='cuda')
  for bi,b in enumerate(bases):
   tb=tangent_raw(b.T)
   for ci in range(8):
    if ci==0:ids=torch.arange(W.shape[0],device='cuda');weights=p[0]
    elif ci<=5:ids=top[:cfg['budgets'][ci-1]];weights=p[0,ids];weights=weights/weights.sum()
    elif ci==6:ids=top[:128];weights=p[0,ids][permutation];weights=weights/weights.sum()
    else:ids=randomids;weights=p[0,top[:128]];weights=weights/weights.sum()
    bb=tb[:,ids];ll=tl[:,ids];bb=bb-(bb*weights).sum(-1,keepdim=True);ll=ll-(ll*weights).sum(-1,keepdim=True);g=.5*(bb*weights)@bb.T;rhs=.5*(bb*weights)@ll.T;reg=g+cfg['ridge']*g.trace()/64*torch.eye(64,device='cuda');c=torch.linalg.solve(reg,rhs);solve=max(solve,float((reg@c-rhs).norm()/rhs.norm()));v=ret[:,si]+c.T@b.T+rad[:,si];res=tf-tangent(v);lg=lp(v);errors[ci,bi,:,si]=torch.stack([.5*(p*res.square()).sum(-1),(pt*(lt-lg)).sum(-1)],-1).cpu().numpy()
  np.savez_compressed(STEM.with_name(STEM.name+'_partial_rows.npz'),errors=errors[:,:,:,:si+1]);print('site',si,'complete',flush=True)
 assert np.isfinite(errors).all();reference=old['errors'][3:8];replay=float(abs(errors[0]-reference).max());np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),errors=errors,masses=np.array(masses));j={'experiment':'H445','config':cfg,'conditions':labels,'full_replay_max_abs':replay,'solve_relative_max':solve,'numerical_gates':{'full_replay':replay<1e-8,'solve':solve<1e-8},'unique_completed_documents':12480};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');print('FINISHED',j['numerical_gates'],flush=True)
if __name__=='__main__':main()

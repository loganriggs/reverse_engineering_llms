"""H443 fixed output spans with baseline-context projection coefficients."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model
from run_bilin18_l17_fisher_ranked_factors import native_drms
STEM=ART/'bilin18_l17_context_projection'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,h in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==h
 old=np.load(ART/'bilin18_l17_operator_transfer_confirmation_rows.npz');projection=np.load(ART/'bilin18_l17_metric_compiled_selection_frozen_projection.npz');selection=json.loads((ART/'bilin18_l17_metric_compiled_selection_selection.json').read_text());enc=torch.tensor(projection['encoders'][selection['selected']],device='cuda');dec=torch.tensor(projection['decoders'][selection['selected']],device='cuda');learned=torch.linalg.qr(dec).Q;random=torch.linalg.qr(torch.tensor(np.random.default_rng(cfg['random_seed']).standard_normal((1152,64)),device='cuda')).Q;model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();L,R,D=[m.weight.detach().double() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];W=model.lm_head.weight.double();pre=torch.tensor(old['pre'],device='cuda').double();st=torch.tensor(old['state'],device='cuda');base=F.rms_norm(st,(1152,));s=(pre.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();x=pre/s;ix=torch.tensor(np.load(ART/'bilin18_l17_residual_factor_rescue_frozen_sets.npz')['sets128'][0],device='cuda');phi=(x@L.T)*(x@R.T);mo=phi@D.T-phi[:,ix]@D[:,ix].T;Q=torch.linalg.qr(model.blocks[17].attn.c_proj.weight.reshape(1152,9,128)[:,6]).Q.double();m=np.load(ART/'bilin18_l17_unmatched_moment_controls_moments.npz');mu=torch.tensor(m['mean'][1],device='cuda');cov=torch.tensor(m['covariance'][1],device='cuda');ev,U=torch.linalg.eigh(cov+mu[:,None]*mu[None,:]);inv=(U*ev.rsqrt())@U.T;V=torch.tensor(np.load(ART/'bilin18_l17_omitted_operator_shared_frozen.npz')['basis'][:,:16],device='cuda');src=torch.tensor(old['sources'],device='cuda');coeff=src@Q@inv@V;H=torch.tensor(np.load(ART/'bilin18_l17_compiled_omitted_maps_weight_compilation.npz')['source_directions'],device='cuda');sc=2*torch.einsum('ar,nr->an',coeff,x@H/1152)/s.T;rad=sc[:,:,None]*mo[None];compiled=torch.tensor(np.load(ART/'bilin18_l17_compiled_omitted_maps_rows.npz')['predictions'][0],device='cuda');linear=compiled-rad;full=torch.tensor(old['full'],device='cuda').double();ret=torch.tensor(old['retained'],device='cuda').double();errors=np.empty((6,24,32,2));normratios=np.empty((6,24,32));local_linear_error=np.empty((6,24,32));audit=0.;condition=[];predictions=np.empty((6,24,32,1152))
 for si in range(32):
  ss=st[si:si+1];th=torch.tanh(base[si:si+1].double()@W.T/30);p=(30*th).softmax(-1);cap=1-th.square()
  def tangent(v):
   dt=(native_drms(ss.double(),v)@W.T)*cap;return dt-(p*dt).sum(-1,keepdim=True)
  def lp(v):return (30*torch.tanh(F.rms_norm(ss+v.float(),(1152,)).double()@W.T/30)).log_softmax(-1)
  tl=tangent(linear[:,si]);guesses=[(linear[:,si]@enc.T)@dec.T,None,(linear[:,si]@random)@random.T,None,None,None]
  for bi,b in enumerate([learned,random]):
   tb=tangent(b.T);g=.5*(tb*p)@tb.T;rhs=.5*(tb*p)@tl.T;evals=torch.linalg.eigvalsh(g);condition.append({'site':si,'basis':bi,'min_eigenvalue':float(evals.min()),'max_eigenvalue':float(evals.max())})
   for ri,ridge in enumerate(cfg['ridge_multipliers']):
    reg=g+ridge*g.trace()/64*torch.eye(64,device='cuda');c=torch.linalg.solve(reg,rhs);audit=max(audit,float((reg@c-rhs).norm()/rhs.norm()));slot=(1 if bi==0 else 3) if ri==0 else (4 if bi==0 else 5);guesses[slot]=c.T@b.T
  tf=tangent(full[:,si]);lt=lp(full[:,si]);pt=lt.exp()
  for method,lin in enumerate(guesses):
   correction=lin+rad[:,si];predictions[method,:,si]=correction.cpu().numpy();v=ret[:,si]+correction;res=tf-tangent(v);lg=lp(v);errors[method,:,si]=torch.stack([.5*(p*res.square()).sum(-1),(pt*(lt-lg)).sum(-1)],-1).cpu().numpy();normratios[method,:,si]=(lin.norm(dim=-1)/linear[:,si].norm(dim=-1).clamp_min(1e-12)).cpu().numpy();lr=tl-tangent(lin);local_linear_error[method,:,si]=(.5*(p*lr.square()).sum(-1)).cpu().numpy()
  print('site',si,'complete',flush=True)
 assert np.isfinite(errors).all();prior=np.load(ART/'bilin18_l17_metric_compiled_selection_rows.npz')['errors'][1];replay=float(abs(errors[0]-prior).max());np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),errors=errors,normratios=normratios,local_linear_errors=local_linear_error,predictions=predictions);j={'experiment':'H443','config':cfg,'methods':['learned_fixed','learned_context','random_fixed','random_context','learned_context_strong_ridge','random_context_strong_ridge'],'solve_relative_max':audit,'fixed_replay_max_abs':replay,'numerical_gates':{'solve':audit<1e-8,'fixed_replay':replay<1e-8},'conditioning':condition,'unique_completed_documents':12480};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');print('FINISHED',j['numerical_gates'],flush=True)
if __name__=='__main__':main()

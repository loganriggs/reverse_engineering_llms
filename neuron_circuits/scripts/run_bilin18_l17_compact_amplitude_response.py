"""H455 signed amplitude test of frozen compact source-response laws."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model
from run_bilin18_l17_fisher_ranked_factors import native_drms
STEM=ART/'bilin18_l17_compact_amplitude_response'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,h in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==h
 old=np.load(ART/'bilin18_l17_compact_operation_confirmation_rows.npz');projection=np.load(ART/'bilin18_l17_metric_compiled_selection_frozen_projection.npz');selection=json.loads((ART/'bilin18_l17_metric_compiled_selection_selection.json').read_text());enc=torch.tensor(projection['encoders'][selection['selected']],device='cuda');dec=torch.tensor(projection['decoders'][selection['selected']],device='cuda');learned=torch.linalg.qr(dec).Q;random=torch.linalg.qr(torch.tensor(np.random.default_rng(cfg['random_seeds'][0]).standard_normal((1152,64)),device='cuda')).Q;model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();L,R,D=[m.weight.detach().double() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];W=model.lm_head.weight.double();pre=torch.tensor(old['pre'],device='cuda').double();st=torch.tensor(old['state'],device='cuda');base=F.rms_norm(st,(1152,));s=(pre.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();x=pre/s;ix=torch.tensor(np.load(ART/'bilin18_l17_residual_factor_rescue_frozen_sets.npz')['sets128'][0],device='cuda');phi=(x@L.T)*(x@R.T);mo=phi@D.T-phi[:,ix]@D[:,ix].T;Q=torch.linalg.qr(model.blocks[17].attn.c_proj.weight.reshape(1152,9,128)[:,6]).Q.double();m=np.load(ART/'bilin18_l17_unmatched_moment_controls_moments.npz');mu=torch.tensor(m['mean'][1],device='cuda');cov=torch.tensor(m['covariance'][1],device='cuda');ev,U=torch.linalg.eigh(cov+mu[:,None]*mu[None,:]);inv=(U*ev.rsqrt())@U.T;V=torch.tensor(np.load(ART/'bilin18_l17_omitted_operator_shared_frozen.npz')['basis'][:,:16],device='cuda');src=torch.tensor(old['sources'],device='cuda');coeff=src@Q@inv@V;H=torch.tensor(np.load(ART/'bilin18_l17_compiled_omitted_maps_weight_compilation.npz')['source_directions'],device='cuda');sc=2*torch.einsum('ar,nr->an',coeff,x@H/1152)/s.T;rad=sc[:,:,None]*mo[None];maps=torch.tensor(np.load(ART/'bilin18_l17_compiled_omitted_maps_weight_compilation.npz')['maps'],device='cuda');compiled=-torch.einsum('ar,roi,ni->ano',coeff,maps,x)/s[None]+rad;linear=compiled-rad;full=torch.tensor(old['full'],device='cuda').double();ret=torch.tensor(old['retained'],device='cuda').double();errors=np.empty((6,24,32,2));normratios=np.empty((6,24,32));local_linear_error=np.empty((6,24,32));audit=0.;condition=[];predictions=np.empty((6,24,32,1152))
 bases=[learned]+[torch.linalg.qr(torch.tensor(np.random.default_rng(seed).standard_normal((1152,64)),device='cuda')).Q for seed in cfg['random_seeds']];targets=[linear]
 for rank in [128,256]:
  z=np.load(ART/f'bilin18_l17_joint_input_regression_rank{rank}_compiled.npz');enc=torch.tensor(z['encoder'],device='cuda');tc=torch.tensor(z['maps'],device='cuda');targets.append(-torch.einsum('ar,roj,nj->ano',coeff,tc,x@enc.T)/s[None])
 targets=torch.stack(targets);cal=np.load(ART/'bilin18_l17_tail_mean_approximation_calibration.npz');sharedmean=torch.tensor(cal['sharedmean'],device='cuda');constant=float(cal['constant']);amplitudes=torch.tensor(cfg['amplitudes'],device='cuda');Lf,Rf,Df=L.float(),R.float(),D.float();pre_f=pre.float();xf=F.rms_norm(pre_f,(1152,));phif=(xf@Lf.T)*(xf@Rf.T);allfull=[];allret=[]
 for amp in cfg['amplitudes']:
  fs=[];rs=[]
  for h in src:
   xt=F.rms_norm(pre_f-amp*h.float(),(1152,));phit=(xt@Lf.T)*(xt@Rf.T);fs.append(phit@Df.T-phif@Df.T);rs.append((phit-phif)[:,ix]@Df[:,ix].T)
  allfull.append(torch.stack(fs));allret.append(torch.stack(rs))
 allfull=torch.stack(allfull).double();allret=torch.stack(allret).double();target_replay=float((allfull[6]-full).norm()/full.norm());assert cfg['amplitudes'][6]==1;errors=np.empty((8,3,9,24,128,2));effects=np.empty((8,24,128,2));audit=0.;condmax=0.
 for si in range(128):
  ss=st[si:si+1];th=torch.tanh(base[si:si+1].double()@W.T/30);p=(30*th).softmax(-1);cap=1-th.square();lp0=(30*th).log_softmax(-1);head=p[0].argsort(descending=True)[:128];mh=p[0,head].sum();mt=1-mh;wh=p[0,head]/mh
  def tangent_raw(v):return (native_drms(ss.double(),v)@W.T)*cap
  def tangent(v):
   t=tangent_raw(v);return t-(p*t).sum(-1,keepdim=True)
  def lp(v):return (30*torch.tanh(F.rms_norm(ss+v.float(),(1152,)).double()@W.T/30)).log_softmax(-1)
  lin=targets[:,:,si].reshape(-1,1152);tl=tangent_raw(lin);dzl=native_drms(ss.double(),lin);ll=tl[:,head];lm=(ll*wh).sum(-1);lc=ll-lm[:,None];dl=lm-dzl@sharedmean;f=allfull[:,:,si].reshape(-1,1152);tf=tangent(f);lt=lp(f);pt=lt.exp();effects[:,:,si]=torch.stack([.5*(p*tf.square()).sum(-1),(pt*(lt-lp0)).sum(-1)],-1).reshape(8,24,2).cpu().numpy()
  for bi,b in enumerate(bases):
   tb=tangent_raw(b.T);dzb=native_drms(ss.double(),b.T);bb=tb[:,head];bm=(bb*wh).sum(-1);bc=bb-bm[:,None];db=bm-dzb@sharedmean;g=mh*.5*(bc*wh)@bc.T+mt*constant/1152*(dzb@dzb.T)+.5*mh*mt*db[:,None]*db[None,:];rhs=mh*.5*(bc*wh)@lc.T+mt*constant/1152*(dzb@dzl.T)+.5*mh*mt*db[:,None]*dl[None,:];ev=torch.linalg.eigvalsh(g);lam=torch.maximum(cfg['ridge']*g.trace()/64,(ev[-1]-cfg['condition_cap']*ev[0])/(cfg['condition_cap']-1));reg=g+lam*torch.eye(64,device='cuda');c=torch.linalg.solve(reg,rhs);audit=max(audit,float((reg@c-rhs).norm()/rhs.norm()));condmax=max(condmax,float((ev[-1]+lam)/(ev[0]+lam)));cor=(c.T@b.T).reshape(3,24,1152)
   for mi in range(3):
    v=(allret[:,:,si]+amplitudes[:,None,None]*(rad[:,si]+cor[mi])[None]).reshape(-1,1152);res=tf-tangent(v);lg=lp(v);errors[:,mi,bi,:,si]=torch.stack([.5*(p*res.square()).sum(-1),(pt*(lt-lg)).sum(-1)],-1).reshape(8,24,2).cpu().numpy()
  np.savez_compressed(STEM.with_name(STEM.name+'_partial_rows.npz'),errors=errors[:,:,:,:,:si+1],effects=effects[:,:,:si+1]);print('site',si,'complete',flush=True)
 assert all(np.isfinite(v).all() for v in [errors,effects]);replay=max(float(abs(errors[6,i]-old['errors'][q]).max()) for i,q in enumerate([1,4,5]));np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),errors=errors,effects=effects);j={'experiment':'H455','config':cfg,'methods':['dense','joint128','joint256'],'solve_relative_max':audit,'condition_max':condmax,'target_replay_relative':target_replay,'unit_replay_max_abs':replay,'numerical_gates':{'solve':audit<1e-8,'condition':condmax<cfg['condition_cap']*(1+1e-8),'target_replay':target_replay<1e-4,'unit_replay':replay<1e-6},'unique_completed_documents':12992};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');print('FINISHED',j['numerical_gates'],flush=True)
if __name__=='__main__':main()

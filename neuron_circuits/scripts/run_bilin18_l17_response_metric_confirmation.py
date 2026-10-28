"""H425 larger fresh factorial confirmation."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources
from run_bilin18_l17_fisher_ranked_factors import native_drms
STEM=ART/'bilin18_l17_response_metric_confirmation'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());op=ART/'bilin18_l17_residual_factor_rescue_frozen_sets.npz';assert hashlib.sha256(op.read_bytes()).hexdigest()==cfg['sets_sha256'];assert hashlib.sha256((ROOT/cfg['corpus']).read_bytes()).hexdigest()==cfg['corpus_sha256'];q=np.load(op);ids=[torch.tensor(v,device='cuda') for v in [q['core'],q['sets128'][0,64:],q['sets128'][0]]];maps=np.stack([np.concatenate([np.random.default_rng(seed+10000*block).permutation(cfg['donor_rows']) for block in range(2)]) for seed in cfg['draw_seeds']]);np.savez_compressed(STEM.with_name(STEM.name+'_donor_maps.npz'),maps=maps);model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();pre,heads,_=collect_sources(model,np.load(ROOT/cfg['corpus']),cfg);assert hashlib.sha256((ROOT/cfg['donor_corpus']).read_bytes()).hexdigest()==cfg['donor_sha256'];_,donor_heads,_=collect_sources(model,np.load(ROOT/cfg['donor_corpus']),dict(cfg,rows=cfg['donor_rows']));S=len(pre);L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];W=model.lm_head.weight.double();xf=F.rms_norm(pre,(1152,));phi0=(xf@L.T)*(xf@R.T);m0=phi0@D.T;state=pre+m0+model.blocks[17].mlp.Down_bias;base=F.rms_norm(state,(1152,));pd=pre.double();Ld,Rd,Dd=L.double(),R.double(),D.double();scale=(pd.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();x=pd/scale;lx=x@Ld.T;rx=x@Rd.T;Q=torch.linalg.qr(model.blocks[17].attn.c_proj.weight.reshape(1152,9,128)[:,6]).Q;axis=(pd@Q.double())@Q.double().T;axis/=axis.norm(dim=-1,keepdim=True);tangent=np.empty((13,3,S));finite=np.empty((13,3,2,S));effect=np.empty((13,S));audit={'clamp_state_error':0.}
 def lp(v):return (30*torch.tanh((v.double()@W.T)/30)).log_softmax(-1)
 oldstem=ART/'bilin18_l17_unmatched_moment_controls';mp=oldstem.with_name(oldstem.name+'_moments.npz');dm=oldstem.with_name(oldstem.name+'_donor_maps.npz');assert hashlib.sha256(mp.read_bytes()).hexdigest()==cfg['moments_sha256'];m=np.load(mp);mu=torch.tensor(m['mean'],device='cuda');root=torch.tensor(m['root'],device='cuda');amps=cfg['amplitudes'];finite=np.empty((len(amps),13,3,S));effect=np.empty((len(amps),13,S));audit={'group_partition_relative_error':0.}
 errors=np.empty((2,2,2,25,3,S));energies=np.empty((2,2,2,25,S));audit={}
 for condition in range(25):
  if condition==0:pay=heads[:,6]
  else:
   family=(condition-1)//12;di=(condition-1)%12
   if family==0:
    sa=(torch.tensor(maps[di],device='cuda')[:,None]*4+torch.arange(4,device='cuda')).reshape(-1);pay=donor_heads[sa,6]
   else:
    noise=torch.tensor(np.random.default_rng(cfg['draw_seeds'][di]+100000).standard_normal((cfg['rows'],4,128)),device='cuda');cc=mu+torch.einsum('npi,pji->npj',noise,root);pay=(cc.reshape(S,128)@Q.double().T).float()
  u=native_drms(pd,-pay.double());dphi=(u@Ld.T)*rx+lx*(u@Rd.T);dfull=dphi@Dd.T;dkept=torch.stack([dphi[:,ix]@Dd[:,ix].T for ix in ids])
  for ai,amp in enumerate(amps):
   xx=F.rms_norm(pre-amp*pay,(1152,));phi1=(xx@L.T)*(xx@R.T);rr=phi1@D.T-m0;delta=phi1-phi0;kept=torch.stack([delta[:,ix]@D[:,ix].T for ix in ids])
   for ri,(full,part) in enumerate([(rr.double(),kept.double()),(amp*dfull,amp*dkept)]):
    if True:
     target=F.rms_norm(state+full.float(),(1152,));pred=F.rms_norm(state[None]+part.float(),(1152,))
    for off in range(0,S,16):
     sl=slice(off,off+16);st=state[sl].double();th=torch.tanh((base[sl].double()@W.T)/30);cap=1-th.square();p=(30*th).softmax(-1)
     def tangent(v):
      dt=(native_drms(st,v)@W.T)*cap;return dt-(p*dt).sum(-1,keepdim=True)
     total=tangent(full[sl]);energies[ri,1,ai,condition,sl]=(.5*(p*total.square()).sum(-1)).cpu().numpy()
     for gi in range(3):
      residual=total-tangent(part[gi,sl]);errors[ri,1,ai,condition,gi,sl]=(.5*(p*residual.square()).sum(-1)).cpu().numpy()
     if True:
      pt=lp(target[sl]);p0=(30*th).log_softmax(-1);energies[ri,0,ai,condition,sl]=(pt.exp()*(pt-p0)).sum(-1).cpu().numpy()
      for gi in range(3):
       pp=lp(pred[gi,sl]);errors[ri,0,ai,condition,gi,sl]=(pt.exp()*(pt-pp)).sum(-1).cpu().numpy()
  np.savez_compressed(STEM.with_name(STEM.name+'_partial_rows.npz'),errors=errors[:,:,:,:condition+1],energies=energies[:,:,:,:condition+1]);print('condition',condition,'complete',flush=True)
 assert np.isfinite(errors).all() and np.isfinite(energies).all();np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),errors=errors,energies=energies);analyze(cfg,errors,energies,audit)
def analyze(cfg,k,e,audit):
 n=cfg['rows'];kd=k.reshape(2,2,2,25,3,n,4).mean(-1);ed=e.reshape(2,2,2,25,n,4).mean(-1)
 def agg(ix,docs):
  kk=kd[:,:,:,ix][:,:,:,:,:,docs].mean((3,5));ee=ed[:,:,:,ix][:,:,:,:,docs].mean((3,4));rec=1-kk/ee[:,:,:,None];return rec,ee,rec[:,:,:,2]-rec[:,:,:,0]-rec[:,:,:,1]
 def all_interactions(docs,draw):
  raw=agg(1+draw,docs);gauss=agg(13+draw,docs);diff=gauss[2]-raw[2];return diff[:,:,1]-diff[:,:,0]
 def contrast(docs,draw):
  ii=all_interactions(docs,draw);return np.array([ii[0,1]-ii[1,1],ii[1,0]-ii[1,1],ii[0,0]-ii[0,1]-ii[1,0]+ii[1,1]])
 docs=np.arange(n);draw=np.arange(12);point=contrast(docs,draw);ii=all_interactions(docs,draw);rng=np.random.default_rng(cfg['bootstrap_seed']);boot=np.stack([contrast(rng.integers(0,n,n),rng.integers(0,12,12)) for _ in range(cfg['bootstrap_samples'])]);rad=np.quantile(abs(boot-point).max(1),.95);ci=np.stack([point-rad,point+rad],-1);fam=[agg(ix,docs) for ix in [[0],np.arange(1,13),np.arange(13,25)]];audit['linear_quadratic_invariance']=float(max(abs(v[2][1,1,1]-v[2][1,1,0]) for v in fam));audit['factorial_closure']=float(abs(point.sum()-(ii[0,0]-ii[1,1])));j={'config':cfg,'response_order':['native','linear'],'metric_order':['exactKL','quadratic'],'families':['actual','raw','gaussian'],'family_recoveries':[v[0].tolist() for v in fam],'family_energies':[v[1].tolist() for v in fam],'family_combination_scores':[v[2].tolist() for v in fam],'distribution_amplitude_interactions':ii.tolist(),'primary_names':['MLP_only','output_only','factorial_interaction'],'primary':point.tolist(),'primary_joint3_ci95':ci.tolist(),'negative02':(ci[:,1]<-.02).tolist(),'equivalence02':(abs(ci)<.02).all(-1).tolist(),'audit':audit,'document_audit':json.loads((ART/'bilin18_response_metric_confirmation_document_audit.json').read_text())};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');np.savez_compressed(STEM.with_name(STEM.name+'_bootstrap.npz'),primary=boot);print('FINISHED',point,ci,audit,flush=True)
if __name__=='__main__':main()

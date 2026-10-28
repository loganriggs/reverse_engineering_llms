"""H421 raw Gaussian moments without geometry matching."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources
from run_bilin18_l17_fisher_ranked_factors import native_drms
STEM=ART/'bilin18_l17_unmatched_moment_controls'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());op=ART/'bilin18_l17_residual_factor_rescue_frozen_sets.npz';assert hashlib.sha256(op.read_bytes()).hexdigest()==cfg['sets_sha256'];assert hashlib.sha256((ROOT/cfg['corpus']).read_bytes()).hexdigest()==cfg['corpus_sha256'];q=np.load(op);ids=[torch.tensor(v,device='cuda') for v in [q['core'],q['sets128'][0,64:],q['sets128'][0]]];maps=np.stack([np.random.default_rng(seed).permutation(cfg['rows']) for seed in cfg['draw_seeds']]);np.savez_compressed(STEM.with_name(STEM.name+'_donor_maps.npz'),maps=maps);model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();pre,heads,_=collect_sources(model,np.load(ROOT/cfg['corpus']),cfg);assert hashlib.sha256((ROOT/cfg['donor_corpus']).read_bytes()).hexdigest()==cfg['donor_sha256'];_,donor_heads,_=collect_sources(model,np.load(ROOT/cfg['donor_corpus']),cfg);S=len(pre);L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];W=model.lm_head.weight.double();xf=F.rms_norm(pre,(1152,));phi0=(xf@L.T)*(xf@R.T);m0=phi0@D.T;state=pre+m0+model.blocks[17].mlp.Down_bias;base=F.rms_norm(state,(1152,));pd=pre.double();Ld,Rd,Dd=L.double(),R.double(),D.double();scale=(pd.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();x=pd/scale;lx=x@Ld.T;rx=x@Rd.T;Q=torch.linalg.qr(model.blocks[17].attn.c_proj.weight.reshape(1152,9,128)[:,6]).Q;axis=(pd@Q.double())@Q.double().T;axis/=axis.norm(dim=-1,keepdim=True);tangent=np.empty((13,3,S));finite=np.empty((13,3,2,S));effect=np.empty((13,S));audit={'clamp_state_error':0.}
 def lp(v):return (30*torch.tanh((v.double()@W.T)/30)).log_softmax(-1)
 # Gaussian calibration is separate from recipients; population covariance matches donor pool.
 coeff=(donor_heads[:,6].double()@Q.double()).reshape(cfg['rows'],4,128);mu=coeff.mean(0);center=coeff-mu;cov=torch.einsum('npi,npj->pij',center,center)/cfg['rows'];ev,V=torch.linalg.eigh(cov);root=V@torch.diag_embed(ev.clamp_min(0).sqrt());audit['covariance_root_error']=float((root@root.transpose(-1,-2)-cov).abs().max());np.savez_compressed(STEM.with_name(STEM.name+'_moments.npz'),mean=mu.cpu().numpy(),covariance=cov.cpu().numpy(),root=root.cpu().numpy());audit['donor_projection_error']=float(((donor_heads[:,6].double()-(coeff.reshape(S,128)@Q.double().T)).norm(dim=-1)/donor_heads[:,6].double().norm(dim=-1)).max())
 for condition in range(13):
  if condition==0:pay=heads[:,6]
  else:
   family=(condition-1)//6;di=(condition-1)%6
   if family==0:
    sa=(torch.tensor(maps[di],device='cuda')[:,None]*4+torch.arange(4,device='cuda')).reshape(-1);pay=donor_heads[sa,6]
   else:
    noise=torch.tensor(np.random.default_rng(cfg['draw_seeds'][di]+100000).standard_normal((cfg['rows'],4,128)),device='cuda');cc=mu+torch.einsum('npi,pji->npj',noise,root);pay=(cc.reshape(S,128)@Q.double().T).float()
  hd=pay.double();u=native_drms(pd,-hd);phi=(u@Ld.T)*rx+lx*(u@Rd.T);vs=[phi@Dd.T]+[phi[:,ix]@Dd[:,ix].T for ix in ids[:2]];par=(hd*axis).sum(-1,keepdim=True)*axis;perp=hd-par;parts=[]
  for v in [par,perp]:
   uu=-v/scale;parts.append((uu@Ld.T)*rx+lx*(uu@Rd.T))
  vs.extend([part[:,ix]@Dd[:,ix].T for ix in ids[:2] for part in parts]);xx=F.rms_norm(pre-pay,(1152,));phi1=(xx@L.T)*(xx@R.T);r=phi1@D.T-m0;delta=phi1-phi0;kept=torch.stack([delta[:,ix]@D[:,ix].T for ix in ids]);target=F.rms_norm(state+r,(1152,));pred=F.rms_norm(state[None]+kept,(1152,));clamp=F.rms_norm(state[None]+r[None]-kept,(1152,))
  for gi,ix in enumerate(ids):
   patch=phi1[:4].clone();patch[:,ix]=phi0[:4,ix];direct=F.rms_norm(pre[:4]+patch@D.T+model.blocks[17].mlp.Down_bias,(1152,));audit['clamp_state_error']=max(audit['clamp_state_error'],float((direct-clamp[gi,:4]).abs().max()))
  for off in range(0,S,16):
   sl=slice(off,off+16);st=state[sl].double();th=torch.tanh((base[sl].double()@W.T)/30);cap=1-th.square();p=(30*th).softmax(-1);ts=[]
   for v in vs:
    dt=(native_drms(st,v[sl])@W.T)*cap;dt-=(p*dt).sum(-1,keepdim=True);ts.append(dt)
   total,c,a,cp,co,ap,ao=ts;tangent[condition,:,sl]=torch.stack([(p*total.square()).sum(-1),2*(p*c*a).sum(-1),2*(p*(cp*ao+co*ap)).sum(-1)]).cpu().numpy();pt=lp(target[sl]);p0=(30*th).log_softmax(-1);effect[condition,sl]=(pt.exp()*(pt-p0)).sum(-1).cpu().numpy()
   for gi in range(3):
    pp=lp(pred[gi,sl]);pc=lp(clamp[gi,sl]);finite[condition,gi,0,sl]=(pt.exp()*(pt-pp)).sum(-1).cpu().numpy();finite[condition,gi,1,sl]=(pc.exp()*(pc-p0)).sum(-1).cpu().numpy()
  np.savez_compressed(STEM.with_name(STEM.name+'_partial_rows.npz'),tangent=tangent[:condition+1],finite=finite[:condition+1],effect=effect[:condition+1]);print('condition',condition,'complete',flush=True)
 assert np.isfinite(tangent).all() and np.isfinite(finite).all();np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),tangent=tangent,finite=finite,effect=effect);STEM.with_name(STEM.name+'_audit.json').write_text(json.dumps(audit,indent=2)+'\n');analyze(cfg,tangent,finite,effect,audit)
def analyze(cfg,t,k,e,audit):
 n=cfg['rows'];td=t.reshape(13,3,n,4).mean(-1);kd=k.reshape(13,3,2,n,4).mean(-1);ed=e.reshape(13,n,4).mean(-1)
 def agg(ix,docs):
  v=td[ix][:,:,docs].mean((0,2));kk=kd[ix][:,:,:,docs].mean((0,3));ee=ed[ix][:,docs].mean();rec=1-kk[:,0]/ee;return np.array([v[1]/v[0],rec[2]-rec[0]-rec[1],rec[2]]),rec,v,ee
 def contrast(docs,draw):return agg(7+draw,docs)[0]-agg(1+draw,docs)[0]
 point=contrast(np.arange(n),np.arange(6));rng=np.random.default_rng(cfg['bootstrap_seed']);boot=np.stack([contrast(rng.integers(0,n,n),rng.integers(0,6,6)) for _ in range(cfg['bootstrap_samples'])]);rad=np.quantile(abs(boot-point).max(1),.95);ci=np.stack([point-rad,point+rad],-1);fam=[agg(ix,np.arange(n)) for ix in [[0],np.arange(1,7),np.arange(7,13)]];names=['local_equivalence05','finite_combination_equivalence05','union_equivalence05'];gates={name:bool(np.all(abs(ci[i])<.05)) for i,name in enumerate(names)};j={'config':cfg,'families':['actual','raw_donors','unmatched_gaussian'],'family_scores':[v[0].tolist() for v in fam],'family_recoveries':[v[1].tolist() for v in fam],'family_tangent_means':[v[2].tolist() for v in fam],'family_effects':[float(v[3]) for v in fam],'primary_names':['Gaussian-minus-raw local cross ratio','Gaussian-minus-raw finite combination score','Gaussian-minus-raw union recovery'],'primary':point.tolist(),'primary_joint3_ci95':ci.tolist(),'gates':gates,'audit':audit};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');np.savez_compressed(STEM.with_name(STEM.name+'_bootstrap.npz'),primary=boot);print('FINISHED',gates,point,flush=True)
if __name__=='__main__':main()

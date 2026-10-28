"""H422 source distribution comparison across edit amplitudes."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources
from run_bilin18_l17_fisher_ranked_factors import native_drms
STEM=ART/'bilin18_l17_moment_amplitude_sweep'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());op=ART/'bilin18_l17_residual_factor_rescue_frozen_sets.npz';assert hashlib.sha256(op.read_bytes()).hexdigest()==cfg['sets_sha256'];assert hashlib.sha256((ROOT/cfg['corpus']).read_bytes()).hexdigest()==cfg['corpus_sha256'];q=np.load(op);ids=[torch.tensor(v,device='cuda') for v in [q['core'],q['sets128'][0,64:],q['sets128'][0]]];maps=np.stack([np.random.default_rng(seed).permutation(cfg['rows']) for seed in cfg['draw_seeds']]);np.savez_compressed(STEM.with_name(STEM.name+'_donor_maps.npz'),maps=maps);model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();pre,heads,_=collect_sources(model,np.load(ROOT/cfg['corpus']),cfg);assert hashlib.sha256((ROOT/cfg['donor_corpus']).read_bytes()).hexdigest()==cfg['donor_sha256'];_,donor_heads,_=collect_sources(model,np.load(ROOT/cfg['donor_corpus']),cfg);S=len(pre);L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];W=model.lm_head.weight.double();xf=F.rms_norm(pre,(1152,));phi0=(xf@L.T)*(xf@R.T);m0=phi0@D.T;state=pre+m0+model.blocks[17].mlp.Down_bias;base=F.rms_norm(state,(1152,));pd=pre.double();Ld,Rd,Dd=L.double(),R.double(),D.double();scale=(pd.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();x=pd/scale;lx=x@Ld.T;rx=x@Rd.T;Q=torch.linalg.qr(model.blocks[17].attn.c_proj.weight.reshape(1152,9,128)[:,6]).Q;axis=(pd@Q.double())@Q.double().T;axis/=axis.norm(dim=-1,keepdim=True);tangent=np.empty((13,3,S));finite=np.empty((13,3,2,S));effect=np.empty((13,S));audit={'clamp_state_error':0.}
 def lp(v):return (30*torch.tanh((v.double()@W.T)/30)).log_softmax(-1)
 oldstem=ART/'bilin18_l17_unmatched_moment_controls';mp=oldstem.with_name(oldstem.name+'_moments.npz');dm=oldstem.with_name(oldstem.name+'_donor_maps.npz');assert hashlib.sha256(mp.read_bytes()).hexdigest()==cfg['moments_sha256'];assert hashlib.sha256(dm.read_bytes()).hexdigest()==cfg['maps_sha256'];assert np.array_equal(maps,np.load(dm)['maps']);m=np.load(mp);mu=torch.tensor(m['mean'],device='cuda');root=torch.tensor(m['root'],device='cuda');amps=cfg['amplitudes'];finite=np.empty((len(amps),13,3,S));effect=np.empty((len(amps),13,S));audit={}
 for condition in range(13):
  if condition==0:pay=heads[:,6]
  else:
   family=(condition-1)//6;di=(condition-1)%6
   if family==0:
    sa=(torch.tensor(maps[di],device='cuda')[:,None]*4+torch.arange(4,device='cuda')).reshape(-1);pay=donor_heads[sa,6]
   else:
    noise=torch.tensor(np.random.default_rng(cfg['draw_seeds'][di]+100000).standard_normal((cfg['rows'],4,128)),device='cuda');cc=mu+torch.einsum('npi,pji->npj',noise,root);pay=(cc.reshape(S,128)@Q.double().T).float()
  for ai,amp in enumerate(amps):
   xx=F.rms_norm(pre-amp*pay,(1152,));phi1=(xx@L.T)*(xx@R.T);rr=phi1@D.T-m0;delta=phi1-phi0;kept=torch.stack([delta[:,ix]@D[:,ix].T for ix in ids]);target=F.rms_norm(state+rr,(1152,));pred=F.rms_norm(state[None]+kept,(1152,))
   for off in range(0,S,16):
    sl=slice(off,off+16);pt=lp(target[sl]);p0=lp(base[sl]);effect[ai,condition,sl]=(pt.exp()*(pt-p0)).sum(-1).cpu().numpy()
    for gi in range(3):
     pp=lp(pred[gi,sl]);finite[ai,condition,gi,sl]=(pt.exp()*(pt-pp)).sum(-1).cpu().numpy()
  np.savez_compressed(STEM.with_name(STEM.name+'_partial_rows.npz'),finite=finite[:,:condition+1],effect=effect[:,:condition+1]);print('condition',condition,'complete',flush=True)
 old=np.load(oldstem.with_name(oldstem.name+'_rows.npz'));audit['unit_prediction_replay_max_abs']=float(abs(finite[1]-old['finite'][:,:,0]).max());audit['unit_effect_replay_max_abs']=float(abs(effect[1]-old['effect']).max());assert np.isfinite(finite).all() and np.isfinite(effect).all();np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),finite=finite,effect=effect);analyze(cfg,finite,effect,audit)
def analyze(cfg,k,e,audit):
 n=cfg['rows'];kd=k.reshape(4,13,3,n,4).mean(-1);ed=e.reshape(4,13,n,4).mean(-1)
 def agg(ix,docs):
  kk=kd[:,ix][:,:,:,docs].mean((1,3));ee=ed[:,ix][:,:,docs].mean((1,2));rec=1-kk/ee[:,None];return rec,ee,rec[:,2]-rec[:,0]-rec[:,1]
 def contrast(docs,draw):
  r=agg(1+draw,docs);g=agg(7+draw,docs);return np.stack([g[2]-r[2],g[0][:,2]-r[0][:,2],np.log(g[1]/r[1])],-1)
 point=contrast(np.arange(n),np.arange(6));rng=np.random.default_rng(cfg['bootstrap_seed']);boot=np.stack([contrast(rng.integers(0,n,n),rng.integers(0,6,6)) for _ in range(cfg['bootstrap_samples'])]);tol=np.array(cfg['tolerances']);rad=np.quantile((abs(boot-point)/tol).max((1,2)),.95);ci=np.stack([point-rad*tol,point+rad*tol],-1);fam=[agg(ix,np.arange(n)) for ix in [[0],np.arange(1,7),np.arange(7,13)]];equiv=(abs(ci)<tol[None,:,None]).all(-1);different=((ci[:,:,0]>tol)|(ci[:,:,1]<-tol));j={'config':cfg,'families':['actual','raw','gaussian'],'family_recoveries':[v[0].tolist() for v in fam],'family_effects':[v[1].tolist() for v in fam],'family_combination_scores':[v[2].tolist() for v in fam],'primary_names':['combination score difference','union recovery difference','log effect ratio'],'primary':point.tolist(),'primary_joint12_ci95':ci.tolist(),'equivalence':equiv.tolist(),'meaningful_difference':different.tolist(),'audit':audit,'unit_replay_pass':bool(max(audit.values())<1e-7)};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');np.savez_compressed(STEM.with_name(STEM.name+'_bootstrap.npz'),primary=boot);print('FINISHED',point,equiv,flush=True)
if __name__=='__main__':main()

"""H426 input RMS versus bilinear curvature."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources
from run_bilin18_l17_fisher_ranked_factors import native_drms
STEM=ART/'bilin18_l17_centered_source_reflection'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());op=ART/'bilin18_l17_residual_factor_rescue_frozen_sets.npz';assert hashlib.sha256(op.read_bytes()).hexdigest()==cfg['sets_sha256'];assert hashlib.sha256((ROOT/cfg['corpus']).read_bytes()).hexdigest()==cfg['corpus_sha256'];q=np.load(op);ids=[torch.tensor(v,device='cuda') for v in [q['core'],q['sets128'][0,64:],q['sets128'][0]]];maps=np.stack([np.concatenate([np.random.default_rng(seed+10000*block).permutation(cfg['donor_rows']) for block in range(1)]) for seed in cfg['draw_seeds']]);np.savez_compressed(STEM.with_name(STEM.name+'_donor_maps.npz'),maps=maps);model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();pre,heads,_=collect_sources(model,np.load(ROOT/cfg['corpus']),cfg);assert hashlib.sha256((ROOT/cfg['donor_corpus']).read_bytes()).hexdigest()==cfg['donor_sha256'];_,donor_heads,_=collect_sources(model,np.load(ROOT/cfg['donor_corpus']),dict(cfg,rows=cfg['donor_rows']));S=len(pre);L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];W=model.lm_head.weight.double();xf=F.rms_norm(pre,(1152,));phi0=(xf@L.T)*(xf@R.T);m0=phi0@D.T;state=pre+m0+model.blocks[17].mlp.Down_bias;base=F.rms_norm(state,(1152,));pd=pre.double();Ld,Rd,Dd=L.double(),R.double(),D.double();scale=(pd.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();x=pd/scale;lx=x@Ld.T;rx=x@Rd.T;Q=torch.linalg.qr(model.blocks[17].attn.c_proj.weight.reshape(1152,9,128)[:,6]).Q;axis=(pd@Q.double())@Q.double().T;axis/=axis.norm(dim=-1,keepdim=True);tangent=np.empty((13,3,S));finite=np.empty((13,3,2,S));effect=np.empty((13,S));audit={'clamp_state_error':0.}
 def lp(v):return (30*torch.tanh((v.double()@W.T)/30)).log_softmax(-1)
 oldstem=ART/'bilin18_l17_unmatched_moment_controls';mp=oldstem.with_name(oldstem.name+'_moments.npz');dm=oldstem.with_name(oldstem.name+'_donor_maps.npz');assert hashlib.sha256(mp.read_bytes()).hexdigest()==cfg['moments_sha256'];m=np.load(mp);mu=torch.tensor(m['mean'],device='cuda');root=torch.tensor(m['root'],device='cuda');amps=cfg['amplitudes'];finite=np.empty((len(amps),13,3,S));effect=np.empty((len(amps),13,S));audit={'group_partition_relative_error':0.}
 meanpay=(mu[None].expand(cfg['rows'],-1,-1).reshape(S,128)@Q.double().T);coeff=np.empty((2,12,2,3,4,S));moment=[];audit={'reflection_center':0.,'coefficient_PSD':0.}
 for family in range(2):
  for di in range(12):
   if family==0:
    sa=(torch.tensor(maps[di],device='cuda')[:,None]*4+torch.arange(4,device='cuda')).reshape(-1);pay=donor_heads[sa,6].double()
   else:
    noise=torch.tensor(np.random.default_rng(cfg['draw_seeds'][di]+100000).standard_normal((cfg['rows'],4,128)),device='cuda');cc=mu+torch.einsum('npi,pji->npj',noise,root);pay=cc.reshape(S,128)@Q.double().T
   reflected=2*meanpay-pay;audit['reflection_center']=max(audit['reflection_center'],float(((pay+reflected)/2-meanpay).norm()/meanpay.norm()))
   centered=((pay-meanpay)@Q.double()).reshape(cfg['rows'],4,128);cm=centered.mean(0);second=torch.einsum('npi,npj->pij',centered,centered)/cfg['rows'];moment.append({'family':family,'draw':di,'mean_offset_norm':float(cm.norm()),'second_moment_root_frobenius':float(second.norm()),'covariance_change_from_pairing_frobenius':float(torch.einsum('pi,pj->pij',cm,cm).norm())})
   for ri,h in enumerate([pay,reflected]):
    du=native_drms(pd,-h);ul=du@Ld.T;ur=du@Rd.T;aa=ul*rx+lx*ur;bb=ul*ur
    fulla=aa@Dd.T;fullb=bb@Dd.T;parta=torch.stack([aa[:,ix]@Dd[:,ix].T for ix in ids]);partb=torch.stack([bb[:,ix]@Dd[:,ix].T for ix in ids])
    for off in range(0,S,16):
     sl=slice(off,off+16);st=state[sl].double();th=torch.tanh((base[sl].double()@W.T)/30);cap=1-th.square();p=(30*th).softmax(-1)
     def tangent(v):
      dt=(native_drms(st,v)@W.T)*cap;return dt-(p*dt).sum(-1,keepdim=True)
     ta=tangent(fulla[sl]);tb=tangent(fullb[sl])
     for channel in range(4):
      va=ta if channel==0 else ta-tangent(parta[channel-1,sl]);vb=tb if channel==0 else tb-tangent(partb[channel-1,sl]);vals=torch.stack([.5*(p*va.square()).sum(-1),(p*va*vb).sum(-1),.5*(p*vb.square()).sum(-1)]);coeff[family,di,ri,:,channel,sl]=vals.cpu().numpy()
   np.savez_compressed(STEM.with_name(STEM.name+'_partial_rows.npz'),coefficients=coeff,completed_family=family,completed_draw=di);print('family',family,'draw',di,'complete',flush=True)
 assert np.isfinite(coeff).all();psd=coeff[:,:,:,0]*coeff[:,:,:,2]-coeff[:,:,:,1]**2/4;sc=abs(coeff[:,:,:,0]*coeff[:,:,:,2])+coeff[:,:,:,1]**2/4;audit['coefficient_PSD']=float(np.max(np.maximum(-psd,0)/np.maximum(sc,1e-30)))
 np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),coefficients=coeff);STEM.with_name(STEM.name+'_audit.json').write_text(json.dumps({'numerical':audit,'moments':moment},indent=2)+'\n');analyze(cfg,coeff,audit,moment)
def analyze(cfg,c,audit,moment):
 n=cfg['rows'];c=c.reshape(2,12,2,3,4,n,4).mean(-1)
 def agg(docs,draw):return c[:,draw][:,:,:,:,:,docs].mean((1,5))
 def contrast(v):
  energy=v[:,:,:,0];num=v[:,:,:,1]+v[:,:,:,2]-v[:,:,:,3]-energy;scale=energy[:,0,0].mean();oddn=(num[:,0,1]-num[:,1,1])/2;odde=(energy[:,0,1]-energy[:,1,1])/2;evenn=num[:,:,2].mean(1);evene=energy[:,:,2].mean(1)
  return np.array([4*np.diff(oddn)[0],4*np.diff(odde)[0],16*np.diff(evenn)[0],16*np.diff(evene)[0]])/scale
 v=agg(np.arange(n),np.arange(12));point=contrast(v);rng=np.random.default_rng(cfg['bootstrap_seed']);boot=np.stack([contrast(agg(rng.integers(0,n,n),rng.integers(0,12,12))) for _ in range(cfg['bootstrap_samples'])]);rad=np.quantile(abs(boot-point).max(1),.95);ci=np.stack([point-rad,point+rad],1)
 def scores(v,a):
  curves=np.einsum('frpc,p->frc',v,np.array([1,a,a*a]));return (curves[:,:,1]+curves[:,:,2]-curves[:,:,3]-curves[:,:,0])/curves[:,:,0]
 j={'experiment':'H429','config':cfg,'primary':point.tolist(),'primary_joint4_ci95':ci.tolist(),'sign_resolved':((ci[:,0]>0)|(ci[:,1]<0)).tolist(),'equivalence02':(abs(ci)<.02).all(1).tolist(),'mean_coefficients':v.tolist(),'combination_scores':{str(a):scores(v,a).tolist() for a in [-4,-1,.25,1,4]},'audit':audit,'moments':moment,'numerical_gates':{'reflection_center':audit['reflection_center']<1e-12,'coefficient_PSD':audit['coefficient_PSD']<1e-8},'document_audit':json.loads((ART/'bilin18_centered_source_reflection_document_audit.json').read_text())}
 STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');np.savez_compressed(STEM.with_name(STEM.name+'_bootstrap.npz'),primary=boot);print('FINISHED',point,ci,audit,flush=True)
if __name__=='__main__':main()

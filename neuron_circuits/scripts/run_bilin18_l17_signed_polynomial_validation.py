"""H428 direct signed/intermediate polynomial validation."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources
from run_bilin18_l17_fisher_ranked_factors import native_drms
STEM=ART/'bilin18_l17_signed_polynomial_validation'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());op=ART/'bilin18_l17_residual_factor_rescue_frozen_sets.npz';assert hashlib.sha256(op.read_bytes()).hexdigest()==cfg['sets_sha256'];assert hashlib.sha256((ROOT/cfg['corpus']).read_bytes()).hexdigest()==cfg['corpus_sha256'];q=np.load(op);ids=[torch.tensor(v,device='cuda') for v in [q['core'],q['sets128'][0,64:],q['sets128'][0]]];maps=np.stack([np.concatenate([np.random.default_rng(seed+10000*block).permutation(cfg['donor_rows']) for block in range(2)]) for seed in cfg['draw_seeds']]);np.savez_compressed(STEM.with_name(STEM.name+'_donor_maps.npz'),maps=maps);model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();pre,heads,_=collect_sources(model,np.load(ROOT/cfg['corpus']),cfg);assert hashlib.sha256((ROOT/cfg['donor_corpus']).read_bytes()).hexdigest()==cfg['donor_sha256'];_,donor_heads,_=collect_sources(model,np.load(ROOT/cfg['donor_corpus']),dict(cfg,rows=cfg['donor_rows']));S=len(pre);L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];W=model.lm_head.weight.double();xf=F.rms_norm(pre,(1152,));phi0=(xf@L.T)*(xf@R.T);m0=phi0@D.T;state=pre+m0+model.blocks[17].mlp.Down_bias;base=F.rms_norm(state,(1152,));pd=pre.double();Ld,Rd,Dd=L.double(),R.double(),D.double();scale=(pd.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();x=pd/scale;lx=x@Ld.T;rx=x@Rd.T;Q=torch.linalg.qr(model.blocks[17].attn.c_proj.weight.reshape(1152,9,128)[:,6]).Q;axis=(pd@Q.double())@Q.double().T;axis/=axis.norm(dim=-1,keepdim=True);tangent=np.empty((13,3,S));finite=np.empty((13,3,2,S));effect=np.empty((13,S));audit={'clamp_state_error':0.}
 def lp(v):return (30*torch.tanh((v.double()@W.T)/30)).log_softmax(-1)
 oldstem=ART/'bilin18_l17_unmatched_moment_controls';mp=oldstem.with_name(oldstem.name+'_moments.npz');dm=oldstem.with_name(oldstem.name+'_donor_maps.npz');assert hashlib.sha256(mp.read_bytes()).hexdigest()==cfg['moments_sha256'];m=np.load(mp);mu=torch.tensor(m['mean'],device='cuda');root=torch.tensor(m['root'],device='cuda');amps=cfg['amplitudes'];finite=np.empty((len(amps),13,3,S));effect=np.empty((len(amps),13,S));audit={'group_partition_relative_error':0.}
 coeffpath=ART/'bilin18_l17_normalization_curvature_factorial_bilinear_polynomial_coefficients.npz';assert hashlib.sha256(coeffpath.read_bytes()).hexdigest()==cfg['coefficients_sha256'];coef=np.load(coeffpath)['coefficients'];fullS=S;S=cfg['validation_documents']*4
 pd=pd[:S];x=x[:S];lx=lx[:S];rx=rx[:S];state=state[:S];base=base[:S]
 measured=np.empty((len(amps),len(cfg['conditions']),4,S));predicted=np.empty_like(measured)
 for ci,condition in enumerate(cfg['conditions']):
  if condition==0:pay=heads[:S,6]
  elif condition<=12:
   di=condition-1;sa=(torch.tensor(maps[di],device='cuda')[:,None]*4+torch.arange(4,device='cuda')).reshape(-1);pay=donor_heads[sa[:S],6]
  else:
   di=condition-13;noise=torch.tensor(np.random.default_rng(cfg['draw_seeds'][di]+100000).standard_normal((cfg['rows'],4,128)),device='cuda');cc=mu+torch.einsum('npi,pji->npj',noise,root);pay=(cc.reshape(fullS,128)@Q.double().T).float()[:S]
  du=native_drms(pd,-pay.double())
  for ai,amp in enumerate(amps):
   # Recompute product at changed input directly, without polynomial expansion.
   changed=x+amp*du;delta=(changed@Ld.T)*(changed@Rd.T)-lx*rx
   full=delta@Dd.T;part=torch.stack([delta[:,ix]@Dd[:,ix].T for ix in ids])
   for off in range(0,S,16):
    sl=slice(off,off+16);st=state[sl].double();th=torch.tanh((base[sl].double()@W.T)/30);cap=1-th.square();p=(30*th).softmax(-1)
    def tangent(v):
     dt=(native_drms(st,v)@W.T)*cap;return dt-(p*dt).sum(-1,keepdim=True)
    total=tangent(full[sl]);measured[ai,ci,0,sl]=(.5*(p*total.square()).sum(-1)).cpu().numpy()
    for gi in range(3):
     residual=total-tangent(part[gi,sl]);measured[ai,ci,gi+1,sl]=(.5*(p*residual.square()).sum(-1)).cpu().numpy()
   predicted[ai,ci]=sum(coef[k,condition,:,:S]*amp**(k+2) for k in range(3))
   print('condition',condition,'amplitude',amp,'complete',flush=True)
  np.savez_compressed(STEM.with_name(STEM.name+'_partial_rows.npz'),measured=measured[:,:ci+1],predicted=predicted[:,:ci+1])
 assert np.isfinite(measured).all() and np.isfinite(predicted).all()
 rel=abs(measured-predicted)/np.maximum(abs(measured),1e-20);pooled=abs(measured.mean(-1)-predicted.mean(-1))/np.maximum(abs(measured.mean(-1)),1e-20)
 def score(t):return (t[:,:,1]+t[:,:,2]-t[:,:,3]-t[:,:,0])/t[:,:,0]
 scoreerr=float(abs(score(measured.mean(-1))-score(predicted.mean(-1))).max())
 gates={'site_energy_relative':float(rel.max())<cfg['site_relative_tolerance'],'pooled_energy_relative':float(pooled.max())<cfg['pooled_relative_tolerance'],'pooled_combination_absolute':scoreerr<cfg['combination_absolute_tolerance']}
 j={'experiment':'H428','config':cfg,'scope':'Numerical validation on first64 reused recipients, actual and first raw/Gaussian draws; new amplitudes, no new documents. Not statistical replication or exact-RMS/KL validation.','site_energy_relative_max':float(rel.max()),'pooled_energy_relative_max':float(pooled.max()),'pooled_combination_absolute_max':scoreerr,'gates':gates,'measured_combination_scores':score(measured.mean(-1)).tolist(),'predicted_combination_scores':score(predicted.mean(-1)).tolist(),'unique_completed_documents':10944}
 np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),measured=measured,predicted=predicted);STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');print('FINISHED',json.dumps(j),flush=True)
if __name__=='__main__':main()

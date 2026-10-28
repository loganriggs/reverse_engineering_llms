"""H444 unused-context frozen-rule transfer and component accounting."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources
from run_bilin18_l17_fisher_ranked_factors import native_drms
STEM=ART/'bilin18_l17_context_projection_transfer'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,h in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==h
 oldcfg=json.loads((ROOT/'configs/bilin18_l17_operator_transfer_confirmation.json').read_text());assert hashlib.sha256((ROOT/oldcfg['corpus']).read_bytes()).hexdigest()==oldcfg['corpus_sha256'];oldsel=json.loads((ART/'bilin18_l17_operator_transfer_confirmation_selection.json').read_text());excluded=np.r_[oldsel['low'],oldsel['median']];remaining=np.setdiff1d(np.arange(512),excluded);sites=np.random.default_rng(cfg['context_seed']).permutation(remaining)[:cfg['context_count']];assert not set(sites)&set(excluded);STEM.with_name(STEM.name+'_selection.json').write_text(json.dumps({'sites':sites.tolist(),'excluded':excluded.tolist(),'scope':'Previously baseline-collected but intervention-unused complete documents within H438 panel; random sample of remaining480. Not new corpus documents.'},indent=2)+'\n')
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();Lf,Rf,Df=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];L,R,D=Lf.double(),Rf.double(),Df.double();W=model.lm_head.weight.double();allpre,_,_=collect_sources(model,np.load(ROOT/oldcfg['corpus']),oldcfg);pre=allpre[sites];xf=F.rms_norm(pre,(1152,));phi=(xf@Lf.T)*(xf@Rf.T);st=pre+phi@Df.T+model.blocks[17].mlp.Down_bias;base=F.rms_norm(st,(1152,));_,dh,_=collect_sources(model,np.load(ROOT/oldcfg['donor_corpus']),oldcfg);Q=torch.linalg.qr(model.blocks[17].attn.c_proj.weight.reshape(1152,9,128)[:,6]).Q.double();m=np.load(ART/'bilin18_l17_unmatched_moment_controls_moments.npz');mu=torch.tensor(m['mean'][1],device='cuda');cov=torch.tensor(m['covariance'][1],device='cuda');gr=torch.tensor(m['root'][1],device='cuda');ev,U=torch.linalg.eigh(cov+mu[:,None]*mu[None,:]);inv=(U*ev.rsqrt())@U.T;V=torch.tensor(np.load(ART/'bilin18_l17_omitted_operator_shared_frozen.npz')['basis'][:,:16],device='cuda');H=torch.tensor(np.load(ART/'bilin18_l17_compiled_omitted_maps_weight_compilation.npz')['source_directions'],device='cuda');T=torch.tensor(np.load(ART/'bilin18_l17_compiled_omitted_maps_weight_compilation.npz')['maps'],device='cuda');sources=[];donors=[]
 for seed in cfg['source_seeds']:
  di=int(np.random.default_rng(seed).integers(512));donors.append(di);sources.append(dh[di,6].double())
 for seed in cfg['source_seeds']:
  noise=torch.tensor(np.random.default_rng(seed+100000).standard_normal(128),device='cuda');sources.append((mu+noise@gr.T)@Q.T)
 sources=torch.stack(sources);coeff=sources@Q@inv@V;ix=torch.tensor(np.load(ART/'bilin18_l17_residual_factor_rescue_frozen_sets.npz')['sets128'][0],device='cuda');s=(pre.double().square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();x=pre.double()/s;phid=(x@L.T)*(x@R.T);mo=phid@D.T-phid[:,ix]@D[:,ix].T;A=torch.einsum('roi,ni->nro',T,x);linear=-torch.einsum('ar,nro->ano',coeff,A)/s[None];rad=2*torch.einsum('ar,nr->an',coeff,x@H/1152)[:,:,None]*mo[None]/s[None];full=[];ret=[]
 for h in sources:
  xt=F.rms_norm(pre-h.float(),(1152,));phit=(xt@Lf.T)*(xt@Rf.T);full.append(phit@Df.T-phi@Df.T);ret.append((phit-phi)[:,ix]@Df[:,ix].T)
 full=torch.stack(full).double();ret=torch.stack(ret).double();pj=np.load(ART/'bilin18_l17_metric_compiled_selection_frozen_projection.npz');sel=json.loads((ART/'bilin18_l17_metric_compiled_selection_selection.json').read_text())['selected'];enc=torch.tensor(pj['encoders'][sel],device='cuda');dec=torch.tensor(pj['decoders'][sel],device='cuda');bases=[torch.linalg.qr(dec).Q]+[torch.linalg.qr(torch.tensor(np.random.default_rng(seed).standard_normal((1152,64)),device='cuda')).Q for seed in cfg['random_seeds']];labels=['union_only','union_radial','learned_fixed','learned_context']+[f'random_context_{i}' for i in range(4)]+['learned_context_no_radial']+[f'random_context_no_radial_{i}' for i in range(4)]+['exact_compiled'];errors=np.empty((len(labels),24,len(sites),2));effects=np.empty((24,len(sites),2));audit=0.
 for si in range(len(sites)):
  ss=st[si:si+1];th=torch.tanh(base[si:si+1].double()@W.T/30);p=(30*th).softmax(-1);cap=1-th.square();lp0=(30*th).log_softmax(-1)
  def tangent(v):
   dt=(native_drms(ss.double(),v)@W.T)*cap;return dt-(p*dt).sum(-1,keepdim=True)
  def lp(v):return (30*torch.tanh(F.rms_norm(ss+v.float(),(1152,)).double()@W.T/30)).log_softmax(-1)
  tl=tangent(linear[:,si]);projected=[]
  for b in bases:
   tb=tangent(b.T);g=.5*(tb*p)@tb.T;rhs=.5*(tb*p)@tl.T;reg=g+cfg['ridge']*g.trace()/64*torch.eye(64,device='cuda');c=torch.linalg.solve(reg,rhs);audit=max(audit,float((reg@c-rhs).norm()/rhs.norm()));projected.append(c.T@b.T)
  zero=torch.zeros_like(rad[:,si]);corrections=[zero,rad[:,si],(linear[:,si]@enc.T)@dec.T+rad[:,si]]+[v+rad[:,si] for v in projected]+projected+[linear[:,si]+rad[:,si]];assert len(corrections)==len(labels);tf=tangent(full[:,si]);lt=lp(full[:,si]);pt=lt.exp();effects[:,si]=torch.stack([.5*(p*tf.square()).sum(-1),(pt*(lt-lp0)).sum(-1)],-1).cpu().numpy()
  for method,c in enumerate(corrections):
   v=ret[:,si]+c;res=tf-tangent(v);lg=lp(v);errors[method,:,si]=torch.stack([.5*(p*res.square()).sum(-1),(pt*(lt-lg)).sum(-1)],-1).cpu().numpy()
  np.savez_compressed(STEM.with_name(STEM.name+'_partial_rows.npz'),errors=errors[:,:,:si+1],effects=effects[:,:si+1]);print('site',si,'complete',flush=True)
 assert np.isfinite(errors).all() and np.isfinite(effects).all();np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),errors=errors,effects=effects,sources=sources.cpu().numpy(),pre=pre.cpu().numpy(),state=st.cpu().numpy(),full=full.cpu().numpy(),retained=ret.cpu().numpy());j={'experiment':'H444','config':cfg,'methods':labels,'sites':sites.tolist(),'donor_ids':donors,'solve_relative_max':audit,'solve_gate':audit<1e-8,'unique_completed_documents':12480};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');print('FINISHED',audit,flush=True)
if __name__=='__main__':main()

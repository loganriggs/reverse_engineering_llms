"""H454 frozen compact-operation transfer to fresh documents."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources
from run_bilin18_l17_fisher_ranked_factors import native_drms
STEM=ART/'bilin18_l17_compact_operation_confirmation'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,h in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==h
 oldcfg=cfg;assert hashlib.sha256((ROOT/cfg['corpus']).read_bytes()).hexdigest()==cfg['corpus_sha256'];sites=np.random.default_rng(cfg['context_seed']).permutation(512)[:cfg['context_count']];STEM.with_name(STEM.name+'_selection.json').write_text(json.dumps({'sites':sites.tolist(),'scope':'Random128 of fresh512 corpus documents, selected before baseline collection; full-document exclusions audited'},indent=2)+'\n')
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();Lf,Rf,Df=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];L,R,D=Lf.double(),Rf.double(),Df.double();W=model.lm_head.weight.double();allpre,_,_=collect_sources(model,np.load(ROOT/oldcfg['corpus']),oldcfg);pre=allpre[sites];xf=F.rms_norm(pre,(1152,));phi=(xf@Lf.T)*(xf@Rf.T);st=pre+phi@Df.T+model.blocks[17].mlp.Down_bias;base=F.rms_norm(st,(1152,));_,dh,_=collect_sources(model,np.load(ROOT/oldcfg['donor_corpus']),oldcfg);Q=torch.linalg.qr(model.blocks[17].attn.c_proj.weight.reshape(1152,9,128)[:,6]).Q.double();m=np.load(ART/'bilin18_l17_unmatched_moment_controls_moments.npz');mu=torch.tensor(m['mean'][1],device='cuda');cov=torch.tensor(m['covariance'][1],device='cuda');gr=torch.tensor(m['root'][1],device='cuda');ev,U=torch.linalg.eigh(cov+mu[:,None]*mu[None,:]);inv=(U*ev.rsqrt())@U.T;V=torch.tensor(np.load(ART/'bilin18_l17_omitted_operator_shared_frozen.npz')['basis'][:,:16],device='cuda');H=torch.tensor(np.load(ART/'bilin18_l17_compiled_omitted_maps_weight_compilation.npz')['source_directions'],device='cuda');T=torch.tensor(np.load(ART/'bilin18_l17_compiled_omitted_maps_weight_compilation.npz')['maps'],device='cuda');sources=[];donors=[]
 for seed in cfg['source_seeds']:
  di=int(np.random.default_rng(seed).integers(512));donors.append(di);sources.append(dh[di,6].double())
 for seed in cfg['source_seeds']:
  noise=torch.tensor(np.random.default_rng(seed+100000).standard_normal(128),device='cuda');sources.append((mu+noise@gr.T)@Q.T)
 sources=torch.stack(sources);coeff=sources@Q@inv@V;ix=torch.tensor(np.load(ART/'bilin18_l17_residual_factor_rescue_frozen_sets.npz')['sets128'][0],device='cuda');s=(pre.double().square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();x=pre.double()/s;phid=(x@L.T)*(x@R.T);mo=phid@D.T-phid[:,ix]@D[:,ix].T;A=torch.einsum('roi,ni->nro',T,x);linear=-torch.einsum('ar,nro->ano',coeff,A)/s[None];rad=2*torch.einsum('ar,nr->an',coeff,x@H/1152)[:,:,None]*mo[None]/s[None];full=[];ret=[]
 for h in sources:
  xt=F.rms_norm(pre-h.float(),(1152,));phit=(xt@Lf.T)*(xt@Rf.T);full.append(phit@Df.T-phi@Df.T);ret.append((phit-phi)[:,ix]@Df[:,ix].T)
 full=torch.stack(full).double();ret=torch.stack(ret).double();pj=np.load(ART/'bilin18_l17_metric_compiled_selection_frozen_projection.npz');sel=json.loads((ART/'bilin18_l17_metric_compiled_selection_selection.json').read_text())['selected'];enc=torch.tensor(pj['encoders'][sel],device='cuda');dec=torch.tensor(pj['decoders'][sel],device='cuda');bases=[torch.linalg.qr(dec).Q]+[torch.linalg.qr(torch.tensor(np.random.default_rng(seed).standard_normal((1152,64)),device='cuda')).Q for seed in cfg['random_seeds']]
 targets=[linear];labels=['dense'];coverage=[];compilation_audit=0.
 for family in ['weight','joint']:
  for rank in cfg['ranks']:
   if family=='weight':z=np.load(ART/f'bilin18_l17_compiled_input_compression_weight_{rank}_compiled.npz');enc=torch.tensor(z['input_basis'],device='cuda').T
   else:z=np.load(ART/f'bilin18_l17_joint_input_regression_rank{rank}_compiled.npz');enc=torch.tensor(z['encoder'],device='cuda')
   tc=torch.tensor(z['maps'],device='cuda');comp=-torch.einsum('ar,roj,nj->ano',coeff,tc,x@enc.T)/s[None];alternate=-torch.einsum('ar,ni,roi->ano',coeff,x,tc@enc)/s[None];compilation_audit=max(compilation_audit,float((comp-alternate).norm()/alternate.norm()));targets.append(comp);labels.append(f'{family}_{rank}');coverage.append(float(1-(comp-linear).square().sum()/linear.square().sum()))
 targets=torch.stack(targets);calinfo=np.load(ART/'bilin18_l17_tail_mean_approximation_calibration.npz');sharedmean=torch.tensor(calinfo['sharedmean'],device='cuda');constant=float(calinfo['constant']);ns=len(sites);nb=len(bases);errors=np.empty((6,nb,24,ns,2));effects=np.empty((24,ns,2));norms=np.empty((6,nb,24,ns));audit=0.;condmax=0.
 for si in range(ns):
  ss=st[si:si+1];th=torch.tanh(base[si:si+1].double()@W.T/30);p=(30*th).softmax(-1);cap=1-th.square();lp0=(30*th).log_softmax(-1);head=p[0].argsort(descending=True)[:128];mh=p[0,head].sum();mt=1-mh;wh=p[0,head]/mh
  def tangent_raw(v):return (native_drms(ss.double(),v)@W.T)*cap
  def tangent(v):
   t=tangent_raw(v);return t-(p*t).sum(-1,keepdim=True)
  def lp(v):return (30*torch.tanh(F.rms_norm(ss+v.float(),(1152,)).double()@W.T/30)).log_softmax(-1)
  lin=targets[:,:,si].reshape(-1,1152);tl=tangent_raw(lin);dzl=native_drms(ss.double(),lin);tf=tangent(full[:,si]);lt=lp(full[:,si]);pt=lt.exp();effects[:,si]=torch.stack([.5*(p*tf.square()).sum(-1),(pt*(lt-lp0)).sum(-1)],-1).cpu().numpy();ll=tl[:,head];lm=(ll*wh).sum(-1);lc=ll-lm[:,None];lmean=dzl@sharedmean
  for bi,b in enumerate(bases):
   tb=tangent_raw(b.T);dzb=native_drms(ss.double(),b.T);bb=tb[:,head];bm=(bb*wh).sum(-1);bc=bb-bm[:,None];db=bm-dzb@sharedmean;dl=lm-lmean;g=mh*.5*(bc*wh)@bc.T+mt*constant/1152*(dzb@dzb.T)+.5*mh*mt*db[:,None]*db[None,:];rhs=mh*.5*(bc*wh)@lc.T+mt*constant/1152*(dzb@dzl.T)+.5*mh*mt*db[:,None]*dl[None,:]
   bfull=tb-(tb*p).sum(-1,keepdim=True);lfull=tl[:24]-(tl[:24]*p).sum(-1,keepdim=True);gf=.5*(bfull*p)@bfull.T;rf=.5*(bfull*p)@lfull.T
   for mode,gg,rr in [('full',gf,rf),('simple',g,rhs)]:
    ev=torch.linalg.eigvalsh(gg);lam=torch.maximum(cfg['ridge']*gg.trace()/64,(ev[-1]-cfg['condition_cap']*ev[0])/(cfg['condition_cap']-1));reg=gg+lam*torch.eye(64,device='cuda');c=torch.linalg.solve(reg,rr);audit=max(audit,float((reg@c-rr).norm()/rr.norm()));condmax=max(condmax,float((ev[-1]+lam)/(ev[0]+lam)));cor=c.T@b.T
    if mode=='full':cor=cor[None];offset=0
    else:cor=cor.reshape(5,24,1152);offset=1
    for ci,cc in enumerate(cor):
     v=ret[:,si]+rad[:,si]+cc;res=tf-tangent(v);lg=lp(v);errors[offset+ci,bi,:,si]=torch.stack([.5*(p*res.square()).sum(-1),(pt*(lt-lg)).sum(-1)],-1).cpu().numpy();norms[offset+ci,bi,:,si]=(cc.norm(dim=-1)/linear[:,si].norm(dim=-1)).cpu().numpy()
  np.savez_compressed(STEM.with_name(STEM.name+'_partial_rows.npz'),errors=errors[:,:,:,:si+1],effects=effects[:,:si+1]);print('site',si,'complete',flush=True)

 assert all(np.isfinite(v).all() for v in [errors,effects,norms]);np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),errors=errors,effects=effects,norms=norms,sources=sources.cpu().numpy(),pre=pre.cpu().numpy(),state=st.cpu().numpy(),full=full.cpu().numpy(),retained=ret.cpu().numpy());j={'experiment':'H454','config':cfg,'conditions':['dense_full_output']+[n+'_simple_output' for n in labels],'linear_response_coverage':coverage,'sites':sites.tolist(),'donor_ids':donors,'compilation_relative_max':compilation_audit,'solve_relative_max':audit,'condition_max':condmax,'numerical_gates':{'compilation':compilation_audit<1e-10,'solve':audit<1e-8,'condition':condmax<cfg['condition_cap']*(1+1e-8)},'unique_completed_documents':12992};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');print('FINISHED',j['numerical_gates'],flush=True)
if __name__=='__main__':main()

"""H388 tail-only discovery covariance."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,B
STEM=ART/'bilin18_l17_tail_only_shared_covariance'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,digest in cfg['source_sha256'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==digest
 assert hashlib.sha256((ROOT/cfg['corpus']).read_bytes()).hexdigest()==cfg['corpus_sha256']
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float()
 assert hashlib.sha256((ROOT/cfg['discovery_corpus']).read_bytes()).hexdigest()==cfg['discovery_corpus_sha256']
 dm=json.loads((ROOT/cfg['discovery_corpus']).with_suffix('.manifest.json').read_text());tm=json.loads((ROOT/cfg['corpus']).with_suffix('.manifest.json').read_text());assert not ({z['document_token_sha256'] for z in dm['records']}&{z['document_token_sha256'] for z in tm['records']})
 dcfg=dict(cfg,rows=cfg['discovery_rows'],corpus=cfg['discovery_corpus']);dp,_,_=collect_sources(model,np.load(ROOT/dcfg['corpus'],mmap_mode='r'),dcfg);dl,dr,dd=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];dx=F.rms_norm(dp,(1152,));da=F.rms_norm(dp+B(dx,dx,dl,dr,dd)+model.blocks[17].mlp.Down_bias,(1152,));dw=model.lm_head.weight.double();prob=torch.cat([(30*torch.tanh((da[i:i+16].double()@dw.T)/30)).softmax(-1) for i in range(0,len(da),16)]);tp=prob.clone();ids=tp.topk(32,dim=-1).indices;tp.scatter_(1,ids,0);dmass=tp.sum(-1);tp/=dmass[:,None];mu=tp@dw;KT=dw.T@(tp.mean(0)[:,None]*dw)-mu.T@mu/len(mu);KT=(KT+KT.T)/2;ev=torch.linalg.eigvalsh(KT);assert float(ev.min())>0
 fullmu=prob@dw;fullK=dw.T@(prob.mean(0)[:,None]*dw)-fullmu.T@fullmu/len(fullmu);oldK=torch.tensor(np.load(ART/'bilin18_l17_probability_axes_finite_frozen_basis.npz')['metric'],device='cuda');fullreplay=float((fullK-oldK).norm()/oldK.norm());assert fullreplay<1e-8
 tracecheck=float(((tp*dw.square().sum(-1)).sum(-1)-mu.square().sum(-1)).mean());assert abs(float(torch.trace(KT))-tracecheck)<1e-9
 discoveryaudit={'discovery_documents':128,'evaluation_documents':512,'disjoint_document_hashes':True,'full_covariance_replay_relative':fullreplay,'tail_covariance_min_eigenvalue':float(ev.min()),'tail_covariance_trace':float(torch.trace(KT)),'mean_discovery_tail_mass':float(dmass.mean())};np.savez_compressed(STEM.with_name(STEM.name+'_frozen_tail_covariance.npz'),metric=KT.cpu().numpy(),eigenvalues=ev.cpu().numpy(),mean_tail_probability=tp.mean(0).cpu().numpy(),expected_tail_unembedding=mu.cpu().numpy());print('tail-only covariance frozen',discoveryaudit,flush=True)
 del dp,da,prob,tp,mu,fullmu,dx
 pre,_,_=collect_sources(model,np.load(ROOT/cfg['corpus'],mmap_mode='r'),cfg)
 L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]]
 x=F.rms_norm(pre,(1152,));state=pre+B(x,x,L,R,D)+model.blocks[17].mlp.Down_bias;baseline=F.rms_norm(state,(1152,))
 W=model.lm_head.weight.double();assert model.lm_head.bias is None
 frozen=np.load(ART/'bilin18_l17_probability_axes_finite_frozen_basis.npz');K=torch.tensor(frozen['metric'],device='cuda');bases=[torch.tensor(frozen['eigenvectors'][:,:64],device='cuda'),torch.tensor(np.load(ART/'bilin18_l17_random_span_specificity_seed20270602_basis.npz')['basis'],device='cuda')];S=len(pre);deltas=[]
 for h in cfg['heads']:
  responses=np.load(ART/f'bilin18_l17_top_token_confirmation_h{h}_responses.npz')['responses'];deltas.extend([(F.rms_norm(state+torch.tensor(v,device='cuda'),(1152,))-baseline).double() for v in responses])
 deltas=torch.stack(deltas);grid=cfg['step_grid'];allbest=[];audits=[];wnorm=W.square().sum(-1)
 for bi,V in enumerate(bases):
  mapped=V.T@W.T;KV=K@V;GK=V.T@KV;KTV=KT@V;GKT=V.T@KTV;kl=np.empty((39,4,2,len(grid),S));stationarity=0.;minrank=[16,64];norms=np.empty((39,4,2,S));decomp=0.;scales=np.empty(S);tailmass=np.empty(S)
  for off in range(0,S,8):
   sl=slice(off,off+8);d=deltas[:,sl].transpose(0,1);raw=baseline[sl].double()@W.T;dlog=d@W.T;bp=(30*torch.tanh(raw/30)).softmax(-1);values,ids=bp.topk(32,dim=-1);hp=torch.zeros_like(bp).scatter_(1,ids,values);pt=bp-hp;q=hp.sum(-1);tt=pt.sum(-1);hp=hp/q[:,None];pt=pt/tt[:,None];tailmass[sl]=tt.cpu().numpy();grams=[];right=[];jmeans=[];dmeans=[]
   for p in [hp,pt]:
    jm=(mapped[None]*p[:,None]).sum(-1);dm=(dlog*p[:,None]).sum(-1);J=mapped[None]-jm[:,:,None];tj=dlog-dm[:,:,None];grams.append((J*p[:,None])@J.transpose(1,2));right.append((tj*p[:,None])@J.transpose(1,2));jmeans.append(jm);dmeans.append(dm)
   dj=jmeans[0]-jmeans[1];dd=dmeans[0]-dmeans[1];gb=(q*tt)[:,None,None]*dj[:,:,None]*dj[:,None,:];bb=(q*tt)[:,None,None]*dd[:,:,None]*dj[:,None,:];gh=q[:,None,None]*grams[0];bh=q[:,None,None]*right[0];gt=tt[:,None,None]*grams[1];bt=tt[:,None,None]*right[1]
   mu=pt@W;alpha=((pt*wnorm).sum(-1)-mu.square().sum(-1))/W.shape[1];assert (alpha>0).all();scales[sl]=alpha.cpu().numpy();gi=(tt*alpha)[:,None,None]*torch.eye(64,device='cuda',dtype=torch.float64)[None];br=(tt*alpha)[:,None,None]*(d@V)
   factor=tt*alpha*W.shape[1]/torch.trace(K);gs=factor[:,None,None]*GK;bs=factor[:,None,None]*(d@KV)
   tailfactor=tt*alpha*W.shape[1]/torch.trace(KT);gst=tailfactor[:,None,None]*GKT;bst=tailfactor[:,None,None]*(d@KTV)
   directj=mapped[None]-(mapped[None]*bp[:,None]).sum(-1,keepdim=True);directt=dlog-(dlog*bp[:,None]).sum(-1,keepdim=True);dg=(directj*bp[:,None])@directj.transpose(1,2);db=(directt*bp[:,None])@directj.transpose(1,2);decomp=max(decomp,float(((gh+gt+gb-dg).norm(dim=(1,2))/dg.norm(dim=(1,2))).max()),float(((bh+bt+bb-db).norm(dim=(1,2))/db.norm(dim=(1,2))).max()))
   targetlp=(30*torch.tanh((raw[:,None]+dlog)/30)).log_softmax(-1);tp=targetlp.exp()
   for mi,(G,Bb) in enumerate([(gh+gt+gb,bh+bt+bb),(gh+gi+gb,bh+br+bb),(gh+gs+gb,bh+bs+bb),(gh+gst+gb,bh+bst+bb)]):
    for ri,rank in enumerate(cfg['ranks']):
     g=G[:,:rank,:rank];val,vec=torch.linalg.eigh((g+g.transpose(1,2))/2);keep=val>val[:,-1:]*cfg['pinv_rtol'];minrank[ri]=min(minrank[ri],int(keep.sum(-1).min()));iv=torch.where(keep,1/val.clamp_min(1e-300),0);b=Bb[:,:,:rank];coef=((b@vec)*iv[:,None])@vec.transpose(1,2);stationarity=max(stationarity,float(((coef@g-b).norm(dim=-1)/b.norm(dim=-1).clamp_min(1e-30)).max()));change=coef@mapped[:rank];norms[:,mi,ri,sl]=(coef.square().sum(-1)/d.square().sum(-1).clamp_min(1e-30)).T.cpu().numpy()
     for ti,t in enumerate(grid):
      lp=(30*torch.tanh((raw[:,None]+t*change)/30)).log_softmax(-1);kl[:,mi,ri,ti,sl]=(tp*(targetlp-lp)).sum(-1).T.cpu().numpy()
   if off%512==0:print('basis',bi,'sites',off+8,'of',S,flush=True)
  kl=kl.reshape(3,13,4,2,len(grid),S);best=kl.min(4);old=np.load(ART/f'bilin18_l17_top_token_confirmation_basis{bi}_rows.npz')['grid_kl'][:,:,0];replay=float(np.max(np.abs(kl[:,:,0]-old)));assert np.isfinite(kl).all() and kl.min()>-1e-10 and stationarity<1e-6 and decomp<1e-9 and replay<1e-9
  oldiso=np.load(ART/f'bilin18_l17_tail_covariance_surrogate_basis{bi}_rows.npz')['grid_kl'][:,:,2];isoreplay=float(np.max(np.abs(kl[:,:,1]-oldiso)));assert isoreplay<1e-9
  np.savez_compressed(STEM.with_name(STEM.name+f'_basis{bi}_rows.npz'),grid_kl=kl,best_kl=best,chosen_index=kl.argmin(4),coefficient_norm_ratio=norms,isotropic_scale=scales,tail_mass=tailmass);allbest.append(best);audits.append({'basis':bi,'normal_equation_max_relative':stationarity,'minimum_retained_ranks':minrank,'decomposition_max_relative':decomp,'h384_full_grid_replay_max_absolute':replay,'h386_isotropic_grid_replay_max_absolute':isoreplay,'mean_tail_mass':float(tailmass.mean()),'max_norm_ratio':float(norms.max())})
 analyze(cfg,np.stack(allbest),audits,kl[:,:,0,0,0],discoveryaudit)
def analyze(cfg,best,audits,base,discoveryaudit):
 n=cfg['rows'];groups=[slice(0,1),slice(1,7),slice(7,13)];errors=best.reshape(2,3,13,4,2,n,4);base=base.reshape(3,13,n,4)
 def contrasts(rec):
  ex,iso,shared,tail=[rec[:,:,:,i] for i in range(4)];return np.stack([ex-tail,tail-shared,tail-iso],3)
 rec=np.stack([1-errors[:,:,g].mean((2,5,6))/base[:,g].mean((1,2,3))[None,:,None,None] for g in groups],2);effect=contrasts(rec);rng=np.random.default_rng(cfg['bootstrap_seed']);be=[]
 for _ in range(cfg['bootstrap_samples']):
  ix=rng.integers(0,n,n);rr=[]
  for g in groups:
   e=errors[:,:,g];b=base[:,g];draw=rng.integers(0,e.shape[2],e.shape[2]);em=e[:,:,draw][:,:,:,:,:,ix].mean((2,5,6));bm=b[:,draw][:,:,ix].mean((1,2,3));rr.append(1-em/bm[None,:,None,None])
  be.append(contrasts(np.stack(rr,2)))
 be=np.stack(be);rad=float(np.quantile(np.max(np.abs(be-effect),axis=(1,2,3,4,5)),.95));ci=np.stack([effect-rad,effect+rad],-1);gates={'tail_shared_within5':bool((ci[:,:,:,0,:,1]<.05).all()),'tail_beats_full_shared':bool((ci[:,:,:,1,:,0]>0).all()),'tail_beats_isotropic':bool((ci[:,:,:,2,:,0]>0).all())};j={'config':cfg,'basis_order':['probability','random'],'fit_order':['exact','isotropic','full_shared','tail_shared'],'contrast_order':['exact_minus_tail_shared','tail_minus_full_shared','tail_minus_isotropic'],'families':['actual','moment','donor'],'recovery':rec.tolist(),'effects':effect.tolist(),'effects_joint108_ci95':ci.tolist(),'discovery_audit':discoveryaudit,'audits':audits,'gates':gates};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');np.savez_compressed(STEM.with_name(STEM.name+'_bootstrap.npz'),effects=be);print('FINISHED',gates,'recovery',rec,flush=True)
if __name__=='__main__':main()

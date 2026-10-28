"""H390 fresh confirmation of analytic response prediction."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l8_execution_alignment import angle_matched_payload
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,B
def drms(v,d):
 scale=(v.square().mean(-1,keepdim=True)+torch.finfo(v.dtype).eps).sqrt();return d/scale-v*(v*d).mean(-1,keepdim=True)/scale.pow(3)
STEM=ART/'bilin18_l17_linear_response_confirmation'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,digest in cfg['source_sha256'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==digest
 assert hashlib.sha256((ROOT/cfg['corpus']).read_bytes()).hexdigest()==cfg['corpus_sha256']
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float()
 pre,heads,_=collect_sources(model,np.load(ROOT/cfg['corpus'],mmap_mode='r'),cfg)
 L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]]
 x=F.rms_norm(pre,(1152,));state=pre+B(x,x,L,R,D)+model.blocks[17].mlp.Down_bias;baseline=F.rms_norm(state,(1152,))
 W=model.lm_head.weight.double();assert model.lm_head.bias is None
 frozen=np.load(ART/'bilin18_l17_probability_axes_finite_frozen_basis.npz');K=torch.tensor(frozen['metric'],device='cuda');bases=[torch.tensor(frozen['eigenvectors'][:,:64],device='cuda'),torch.tensor(np.load(ART/'bilin18_l17_random_span_specificity_seed20270602_basis.npz')['basis'],device='cuda')];S=len(pre);deltas=[]
 N=cfg['rows'];linear=[];geometry=0.;replay=0.;fdaudit=0.;native0=B(x,x,L,R,D)
 moments=np.load(ART/'bilin18_l17_source_moment_controls_moments.npz');means=[torch.tensor(v,device='cuda') for v in moments['means']];roots=[torch.tensor(v,device='cuda') for v in moments['roots']];sources=[torch.linalg.qr(model.blocks[17].attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q for h in range(9)];ld,rd,dd=L.double(),R.double(),D.double();eps=torch.finfo(pre.dtype).eps
 for hi,h in enumerate(cfg['heads']):
  responses=[];predictions=[];variants=[('actual',None)]+[(f'moment{i}',seed) for i,seed in enumerate(cfg['draw_seeds'])]+[(f'donor{i}',shift) for i,shift in enumerate(cfg['donor_shifts'])]
  for ci,(name,seed) in enumerate(variants):
   ref=heads[:,h]
   if seed is None:pay=ref
   elif name.startswith('moment'):
    gen=torch.Generator(device='cuda').manual_seed(seed);noise=torch.randn(9,N,4,128,generator=gen,device='cuda',dtype=torch.float64);direction=torch.einsum('npi,pji->npj',noise[h],roots[h])+means[h];pay=angle_matched_payload(pre,ref,sources[h],direction.reshape(S,128).float())
   else:
    site=(((torch.arange(N,device='cuda')+seed)%N)[:,None]*4+torch.arange(4,device='cuda')).reshape(-1);pay=angle_matched_payload(pre,ref,sources[h],heads[site,h]@sources[h])
   geometry=max(geometry,float(((pay.norm(dim=1)-ref.norm(dim=1)).abs()/ref.norm(dim=1)).max()),float(((pre*(pay-ref)).sum(1).abs()/(pre.norm(dim=1)*ref.norm(dim=1))).max()));dx=drms(pre,-pay);lpred=drms(state,B(x,dx,L,R,D)+B(dx,x,L,R,D));linear.append(lpred.double());predictions.append(lpred.cpu().numpy())
   response=B(F.rms_norm(pre-pay,(1152,)),F.rms_norm(pre-pay,(1152,)),L,R,D)-native0;responses.append(response.cpu().numpy());deltas.append((F.rms_norm(state+response,(1152,))-baseline).double())
   p=pre[:4].double();pa=pay[:4].double();bias=model.blocks[17].mlp.Down_bias.double()
   def fn(t):
    z=F.rms_norm(p-t*pa,(1152,),eps=eps);return F.rms_norm(p+B(z,z,ld,rd,dd)+bias,(1152,),eps=eps)
   fd=(fn(1e-4)-fn(-1e-4))/(2e-4);fdaudit=max(fdaudit,float(((fd-lpred[:4].double()).norm(dim=-1)/fd.norm(dim=-1).clamp_min(1e-30)).max()))
  np.savez_compressed(STEM.with_name(STEM.name+f'_h{h}_predictions.npz'),linear_post_rms=np.stack(predictions),native_responses=np.stack(responses));print('linear responses head',h,'audit',fdaudit,flush=True)
 assert geometry<1e-4 and fdaudit<1e-4
 deltas=torch.stack(deltas);linear=torch.stack(linear);grid=cfg['step_grid'];allerrors=[];audits=[];full=np.empty((39,S));base=np.empty((39,S))
 for bi,V in enumerate(bases):
  mapped=V.T@W.T;kl=np.empty((39,2,len(grid),S));pkl=np.empty_like(kl);okl=np.empty_like(kl);stationarity=0.;minrank=[16,64];norms=np.empty((39,2,S))
  for off in range(0,S,8):
   sl=slice(off,off+8);d=deltas[:,sl].transpose(0,1);pred=linear[:,sl].transpose(0,1);raw=baseline[sl].double()@W.T;dlog=d@W.T;plog=pred@W.T;bp=(30*torch.tanh(raw/30)).softmax(-1);targetlp=(30*torch.tanh((raw[:,None]+dlog)/30)).log_softmax(-1);tp=targetlp.exp();predlp=(30*torch.tanh((raw[:,None]+plog)/30)).log_softmax(-1);pp=predlp.exp();full[:,sl]=(tp*(targetlp-predlp)).sum(-1).T.cpu().numpy();base[:,sl]=(tp*(targetlp-(30*torch.tanh(raw/30)).log_softmax(-1)[:,None])).sum(-1).T.cpu().numpy();J=mapped[None]-(mapped[None]*bp[:,None]).sum(-1,keepdim=True);G=(J*bp[:,None])@J.transpose(1,2);tj=plog-(plog*bp[:,None]).sum(-1,keepdim=True);rhs=(tj*bp[:,None])@J.transpose(1,2);exacttj=dlog-(dlog*bp[:,None]).sum(-1,keepdim=True);exactrhs=(exacttj*bp[:,None])@J.transpose(1,2)
   for ri,rank in enumerate(cfg['ranks']):
    g=G[:,:rank,:rank];val,vec=torch.linalg.eigh((g+g.transpose(1,2))/2);keep=val>val[:,-1:]*cfg['pinv_rtol'];minrank[ri]=min(minrank[ri],int(keep.sum(-1).min()));iv=torch.where(keep,1/val.clamp_min(1e-300),0);b=rhs[:,:,:rank];coef=((b@vec)*iv[:,None])@vec.transpose(1,2);stationarity=max(stationarity,float(((coef@g-b).norm(dim=-1)/b.norm(dim=-1).clamp_min(1e-30)).max()));change=coef@mapped[:rank];norms[:,ri,sl]=(coef.square().sum(-1)/pred.square().sum(-1).clamp_min(1e-30)).T.cpu().numpy()
    exactb=exactrhs[:,:,:rank];exactcoef=((exactb@vec)*iv[:,None])@vec.transpose(1,2);exactchange=exactcoef@mapped[:rank];stationarity=max(stationarity,float(((exactcoef@g-exactb).norm(dim=-1)/exactb.norm(dim=-1).clamp_min(1e-30)).max()))
    for gi,t in enumerate(grid):
     lp=(30*torch.tanh((raw[:,None]+t*change)/30)).log_softmax(-1);kl[:,ri,gi,sl]=(tp*(targetlp-lp)).sum(-1).T.cpu().numpy();pkl[:,ri,gi,sl]=(pp*(predlp-lp)).sum(-1).T.cpu().numpy();olp=(30*torch.tanh((raw[:,None]+t*exactchange)/30)).log_softmax(-1);okl[:,ri,gi,sl]=(tp*(targetlp-olp)).sum(-1).T.cpu().numpy()
   if off%512==0:print('basis',bi,'sites',off+8,'of',S,flush=True)
  kl=kl.reshape(3,13,2,len(grid),S);pkl=pkl.reshape(kl.shape);choice=pkl.argmin(3);chosen=np.take_along_axis(kl,choice[:,:,:,None,:],axis=3).squeeze(3);okl=okl.reshape(kl.shape);oracle=okl.min(3);errs=np.stack([oracle,kl.min(3),chosen,kl[:,:,:,grid.index(1.)]],2);assert np.isfinite(kl).all() and kl.min()>-1e-10 and np.isfinite(pkl).all() and pkl.min()>-1e-10 and np.isfinite(okl).all() and okl.min()>-1e-10 and stationarity<1e-6
  np.savez_compressed(STEM.with_name(STEM.name+f'_basis{bi}_rows.npz'),true_grid_kl=kl,exact_oracle_grid_kl=okl,predicted_grid_kl=pkl,predicted_chosen_index=choice,errors=errs,coefficient_norm_ratio=norms);allerrors.append(errs);audits.append({'basis':bi,'normal_equation_max_relative':stationarity,'minimum_retained_ranks':minrank})
 np.savez_compressed(STEM.with_name(STEM.name+'_full_rows.npz'),full_linear_kl=full.reshape(3,13,S),baseline_kl=base.reshape(3,13,S));analyze(cfg,np.stack(allerrors),full.reshape(3,13,S),base.reshape(3,13,S),audits,{'payload_geometry_max':geometry,'central_difference_linear_derivative_max_relative':fdaudit,'central_difference_sites_per_condition':4,'central_difference_step':1e-4})
def analyze(cfg,errors,full,base,audits,predictionaudit):
 n=cfg['rows'];groups=[slice(0,1),slice(1,7),slice(7,13)];errors=errors.reshape(2,3,13,4,2,n,4);full=full.reshape(3,13,n,4);base=base.reshape(3,13,n,4)
 def contrasts(rec):return rec[:,:,:,0]-rec[:,:,:,2],rec[:,:,:,1]-rec[:,:,:,2]
 rec=np.stack([1-errors[:,:,g].mean((2,5,6))/base[:,g].mean((1,2,3))[None,:,None,None] for g in groups],2);fr=np.stack([1-full[:,g].mean((1,2,3))/base[:,g].mean((1,2,3)) for g in groups],1);loss,step=contrasts(rec);rng=np.random.default_rng(cfg['bootstrap_seed']);bl=[];bs=[];bf=[]
 for _ in range(cfg['bootstrap_samples']):
  ix=rng.integers(0,n,n);rr=[];ff=[]
  for g in groups:
   e=errors[:,:,g];b=base[:,g];f=full[:,g];draw=rng.integers(0,e.shape[2],e.shape[2]);em=e[:,:,draw][:,:,:,:,:,ix].mean((2,5,6));bm=b[:,draw][:,:,ix].mean((1,2,3));rr.append(1-em/bm[None,:,None,None]);ff.append(1-f[:,draw][:,:,ix].mean((1,2,3))/bm)
  ll,ss=contrasts(np.stack(rr,2));bl.append(ll);bs.append(ss);bf.append(np.stack(ff,1))
 bl,bs,bf=map(np.stack,[bl,bs,bf]);deviations=np.concatenate([(bl-loss).reshape(len(bl),-1),(bs-step).reshape(len(bs),-1),(bf-fr).reshape(len(bf),-1)],1);rad=float(np.quantile(np.abs(deviations).max(1),.95));lc=np.stack([loss-rad,loss+rad],-1);sc=np.stack([step-rad,step+rad],-1);fc=np.stack([fr-rad,fr+rad],-1);gates={'predicted_fit_within5':bool((lc[...,1]<.05).all()),'full_linear90':bool((fc[...,0]>.9).all()),'step_selection_within5':bool((sc[...,1]<.05).all())};j={'config':cfg,'basis_order':['probability','random'],'fit_order':['exact_response_oracle','linear_fit_true_target_guard','linear_fit_predicted_target_guard','linear_fit_unit_step'],'families':['actual','moment','donor'],'recovery':rec.tolist(),'full_linear_recovery':fr.tolist(),'oracle_minus_predicted_guard':loss.tolist(),'true_guard_minus_predicted_guard':step.tolist(),'loss_joint81_ci95':lc.tolist(),'step_joint81_ci95':sc.tolist(),'full_linear_joint81_ci95':fc.tolist(),'audits':audits,'prediction_audit':predictionaudit,'gates':gates};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');np.savez_compressed(STEM.with_name(STEM.name+'_bootstrap.npz'),loss=bl,step=bs,full=bf);print('FINISHED',gates,'full',fr,'recovery',rec,flush=True)
if __name__=='__main__':main()

"""H374 per-response Fisher oracle in frozen output subspaces."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,B
STEM=ART/'bilin18_l17_probability_axes_oracle'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,digest in cfg['source_sha256'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==digest
 assert hashlib.sha256((ROOT/cfg['corpus']).read_bytes()).hexdigest()==cfg['corpus_sha256']
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float()
 pre,_,_=collect_sources(model,np.load(ROOT/cfg['corpus'],mmap_mode='r'),cfg)
 L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]]
 x=F.rms_norm(pre,(1152,));state=pre+B(x,x,L,R,D)+model.blocks[17].mlp.Down_bias;baseline=F.rms_norm(state,(1152,))
 W=model.lm_head.weight.double();assert model.lm_head.bias is None
 V=torch.tensor(np.load(ART/'bilin18_l17_probability_axes_finite_frozen_basis.npz')['eigenvectors'][:,:64],device='cuda');mapped=V.T@W.T;S=len(pre);deltas=[]
 for h in cfg['heads']:
  responses=np.load(ART/f'bilin18_l17_shared_geometry_precision_h{h}_responses.npz')['responses'];deltas.extend([(F.rms_norm(state+torch.tensor(v,device='cuda'),(1152,))-baseline).double() for v in responses])
 deltas=torch.stack(deltas);kl=np.empty((39,5,S));local=np.empty_like(kl);coefficients=np.zeros((39,2,S,64));normratio=np.empty((39,2,S));minrank=[16,64];stationarity=0.;nonoptimal=0.
 for off in range(0,S,8):
  sl=slice(off,off+8);d=deltas[:,sl].transpose(0,1);raw=baseline[sl].double()@W.T;dlog=d@W.T;bp=(30*torch.tanh(raw/30)).softmax(-1);cap=1-torch.tanh(raw/30).square();J=mapped[None]*cap[:,None];J=J-(J*bp[:,None]).sum(-1,keepdim=True);G=(J*bp[:,None])@J.transpose(1,2);targetj=dlog*cap[:,None];targetj=targetj-(targetj*bp[:,None]).sum(-1,keepdim=True);rhs=(targetj*bp[:,None])@J.transpose(1,2);euclidean=d@V;oracle=[]
  for ri,rank in enumerate(cfg['ranks']):
   g=G[:,:rank,:rank];val,vec=torch.linalg.eigh((g+g.transpose(1,2))/2);keep=val>val[:,-1:]*cfg['pinv_rtol'];minrank[ri]=min(minrank[ri],int(keep.sum(-1).min()));inv=torch.where(keep,1/val.clamp_min(1e-300),0);b=rhs[:,:,:rank];coef=((b@vec)*inv[:,None])@vec.transpose(1,2);oracle.append(coef);res=(coef@g-b).norm(dim=-1)/b.norm(dim=-1).clamp_min(1e-30);stationarity=max(stationarity,float(res.max()));coefficients[:,ri,sl,:rank]=coef.transpose(0,1).cpu().numpy();normratio[:,ri,sl]=(coef.square().sum(-1)/d.square().sum(-1)).T.cpu().numpy()
  changes=[torch.zeros_like(dlog),euclidean[:,:,:16]@mapped[:16],euclidean@mapped,oracle[0]@mapped[:16],oracle[1]@mapped];targetlp=(30*torch.tanh((raw[:,None]+dlog)/30)).log_softmax(-1);tp=targetlp.exp()
  for pi,change in enumerate(changes):
   pred=(30*torch.tanh((raw[:,None]+change)/30)).log_softmax(-1);kl[:,pi,sl]=(tp*(targetlp-pred)).sum(-1).T.cpu().numpy();err=(dlog-change)*cap[:,None];err-= (err*bp[:,None]).sum(-1,keepdim=True);local[:,pi,sl]=(bp[:,None]*err.square()).sum(-1).T.cpu().numpy()
  nonoptimal=max(nonoptimal,float(np.max((local[:,3:5,sl]-local[:,1:3,sl])/local[:,:1,sl])))
  if off%256==0:print('sites',off+8,'of',S,flush=True)
 kl=kl.reshape(3,13,5,S);local=local.reshape(3,13,5,S);old=np.load(ART/'bilin18_l17_probability_axes_finite_rows.npz');replay=max(float(np.max(np.abs(kl[:,:,:3]-old['kl'][:,:,[0,2,3]]))/np.max(old['kl'][:,:,[0,2,3]])),float(np.max(np.abs(local[:,:,:3]-old['local_error'][:,:,[0,2,3]]))/np.max(old['local_error'][:,:,[0,2,3]])))
 assert np.isfinite(kl).all() and kl.min()>-1e-10 and nonoptimal<1e-7 and replay<1e-8 and stationarity<1e-7
 np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),kl=kl,local_error=local,oracle_coefficients=coefficients,oracle_norm_squared_ratio=normratio)
 analyze(cfg,kl,local,{'euclidean_h373_replay_relative':replay,'max_local_suboptimality_relative_to_baseline':nonoptimal,'max_relative_normal_equation_residual':stationarity,'minimum_retained_gram_ranks':minrank,'oracle_norm_ratio_quantiles':np.quantile(np.sqrt(normratio),[.5,.95,.99,1]).tolist()})
def analyze(cfg,kl,local,audit):
 groups=[slice(0,1),slice(1,7),slice(7,13)];kv=kl.reshape(3,13,5,cfg['rows'],4);lv=local.reshape(kv.shape)
 def stats(v):
  m=np.stack([v[:,g].mean((1,3,4)) for g in groups],1);return m,1-m[:,:,1:]/m[:,:,:1]
 km,rec=stats(kv);lm,lr=stats(lv);rng=np.random.default_rng(cfg['bootstrap_seed']);bs=[];bl=[]
 for _ in range(cfg['bootstrap_samples']):
  ix=rng.integers(0,cfg['rows'],cfg['rows']);kk=[];ll=[]
  for g in groups:
   k=kv[:,g];l=lv[:,g];rr=rng.integers(0,k.shape[1],k.shape[1]);kk.append(k[:,rr][:,:,:,ix].mean((1,3,4)));ll.append(l[:,rr][:,:,:,ix].mean((1,3,4)))
  k=np.stack(kk,1);l=np.stack(ll,1);bs.append(1-k[:,:,1:]/k[:,:,:1]);bl.append(1-l[:,:,1:]/l[:,:,:1])
 def band(b,t):
  b=np.stack(b);rad=float(np.quantile(np.max(np.abs(b-t),axis=tuple(range(1,b.ndim))),.95));return np.stack([t-rad,t+rad],-1)
 bs=np.stack(bs);bl=np.stack(bl);ci=band(bs,rec);lci=band(bl,lr);gain=rec[:,:,3]-rec[:,:,1];gci=band(bs[:,:,:,3]-bs[:,:,:,1],gain);gates={'oracle64_local_sufficient':bool((lci[:,:,3,0]>=.8).all()),'oracle64_finite_sufficient':bool((ci[:,:,3,0]>=.8).all()),'oracle64_finite_gain':bool((gci[:,:,0]>.2).all())}
 j={'config':cfg,'families':['actual','moment','donor'],'patches':['baseline','euclidean16','euclidean64','oracle16','oracle64'],'mean_kl':km.tolist(),'recovery':rec.tolist(),'local_recovery':lr.tolist(),'finite_joint36_ci95':ci.tolist(),'local_joint36_ci95':lci.tolist(),'oracle64_gain':gain.tolist(),'oracle64_gain_joint9_ci95':gci.tolist(),'audit':audit,'gates':gates};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');np.savez_compressed(STEM.with_name(STEM.name+'_bootstrap.npz'),finite=bs,local=bl);print('FINISHED',gates,'finite',rec,'local',lr,flush=True)
if __name__=='__main__':main()

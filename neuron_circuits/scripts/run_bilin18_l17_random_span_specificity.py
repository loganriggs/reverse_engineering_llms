"""H377 fixed random-span specificity control."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,B
STEM=ART/'bilin18_l17_random_span_specificity'
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
 results=[];audits=[]
 for seed in cfg['random_seeds']:
  result,audit=run_one(cfg,seed,model,pre,state,baseline,W);results.append(result);audits.append(audit);print('seed',seed,'complete',flush=True)
 analyze(cfg,np.stack(results),audits)
@torch.inference_mode()
def run_one(cfg,seed,model,pre,state,baseline,W):
 gen=torch.Generator(device='cpu').manual_seed(seed);q,rr=torch.linalg.qr(torch.randn(1152,64,generator=gen,dtype=torch.float64));q=q*torch.sign(torch.diag(rr))[None];V=q.to('cuda');mapped=V.T@W.T;S=len(pre);deltas=[]
 np.savez_compressed(STEM.with_name(STEM.name+f'_seed{seed}_basis.npz'),basis=q.numpy())
 for h in cfg['heads']:
  responses=np.load(ART/f'bilin18_l17_shared_geometry_precision_h{h}_responses.npz')['responses'];deltas.extend([(F.rms_norm(state+torch.tensor(v,device='cuda'),(1152,))-baseline).double() for v in responses])
 deltas=torch.stack(deltas);kl=np.empty((39,5,S));local=np.empty_like(kl);coefficients=np.zeros((39,2,S,64));normratio=np.empty((39,2,S));minrank=[16,64];stationarity=0.;nonoptimal=0.
 grid=cfg['step_grid'];gridkl=np.empty((39,2,len(grid),S))
 for off in range(0,S,8):
  sl=slice(off,off+8);d=deltas[:,sl].transpose(0,1);raw=baseline[sl].double()@W.T;dlog=d@W.T;bp=(30*torch.tanh(raw/30)).softmax(-1);cap=1-torch.tanh(raw/30).square();J=mapped[None]*cap[:,None];J=J-(J*bp[:,None]).sum(-1,keepdim=True);G=(J*bp[:,None])@J.transpose(1,2);targetj=dlog*cap[:,None];targetj=targetj-(targetj*bp[:,None]).sum(-1,keepdim=True);rhs=(targetj*bp[:,None])@J.transpose(1,2);euclidean=d@V;oracle=[]
  for ri,rank in enumerate(cfg['ranks']):
   g=G[:,:rank,:rank];val,vec=torch.linalg.eigh((g+g.transpose(1,2))/2);keep=val>val[:,-1:]*cfg['pinv_rtol'];minrank[ri]=min(minrank[ri],int(keep.sum(-1).min()));inv=torch.where(keep,1/val.clamp_min(1e-300),0);b=rhs[:,:,:rank];coef=((b@vec)*inv[:,None])@vec.transpose(1,2);oracle.append(coef);res=(coef@g-b).norm(dim=-1)/b.norm(dim=-1).clamp_min(1e-30);stationarity=max(stationarity,float(res.max()));coefficients[:,ri,sl,:rank]=coef.transpose(0,1).cpu().numpy();normratio[:,ri,sl]=(coef.square().sum(-1)/d.square().sum(-1)).T.cpu().numpy()
  changes=[torch.zeros_like(dlog),euclidean[:,:,:16]@mapped[:16],euclidean@mapped,oracle[0]@mapped[:16],oracle[1]@mapped];targetlp=(30*torch.tanh((raw[:,None]+dlog)/30)).log_softmax(-1);tp=targetlp.exp()
  for pi,change in enumerate(changes):
   pred=(30*torch.tanh((raw[:,None]+change)/30)).log_softmax(-1);kl[:,pi,sl]=(tp*(targetlp-pred)).sum(-1).T.cpu().numpy();err=(dlog-change)*cap[:,None];err-= (err*bp[:,None]).sum(-1,keepdim=True);local[:,pi,sl]=(bp[:,None]*err.square()).sum(-1).T.cpu().numpy()
  for ri,change in enumerate(changes[3:]):
   for gi,t in enumerate(grid):
    pred=(30*torch.tanh((raw[:,None]+t*change)/30)).log_softmax(-1);gridkl[:,ri,gi,sl]=(tp*(targetlp-pred)).sum(-1).T.cpu().numpy()
  nonoptimal=max(nonoptimal,float(np.max((local[:,3:5,sl]-local[:,1:3,sl])/local[:,:1,sl])))
  if off%256==0:print('sites',off+8,'of',S,flush=True)
 kl=kl.reshape(3,13,5,S);local=local.reshape(3,13,5,S);gridkl=gridkl.reshape(3,13,2,len(grid),S);best=gridkl.min(3);old=np.load(ART/'bilin18_l17_probability_axes_oracle_rows.npz');replay=float(np.max(np.abs(kl[:,:,0]-old['kl'][:,:,0]))/np.max(old['kl'][:,:,0]));assert replay<1e-8 and np.isfinite(gridkl).all() and gridkl.min()>-1e-10 and nonoptimal<1e-7 and stationarity<1e-7
 assert np.max(best-kl[:,:,:1])<1e-10 and np.max(best-kl[:,:,[3,4]])<1e-10
 np.savez_compressed(STEM.with_name(STEM.name+f'_seed{seed}_rows.npz'),kl=kl,local_error=local,grid_kl=gridkl,best_kl=best,chosen_index=gridkl.argmin(3),oracle_coefficients=coefficients)
 return best,{'seed':seed,'baseline_replay_relative':replay,'max_local_suboptimality':nonoptimal,'normal_equation_residual':stationarity,'minimum_retained_ranks':minrank,'norm_ratio_quantiles':np.quantile(np.sqrt(normratio),[.5,.95,.99,1]).tolist()}

def analyze(cfg,random,audits):
 groups=[slice(0,1),slice(1,7),slice(7,13)];n=cfg['rows'];prob=np.load(ART/'bilin18_l17_oracle_finite_safeguard_rows.npz')['best_kl'];base=np.load(ART/'bilin18_l17_probability_axes_oracle_rows.npz')['kl'][:,:,0];rv=random.mean(0).reshape(3,13,2,n,4);pv=prob.reshape(rv.shape);bv=base.reshape(3,13,n,4)
 def means(v):return np.stack([v[:,g].mean((1,)+tuple(range(v.ndim-2,v.ndim))) for g in groups],1)
 bm=means(bv);pr=1-means(pv)/bm[:,:,None];rr=1-means(rv)/bm[:,:,None];gap=pr-rr;individual=[(1-means(x.reshape(rv.shape))/bm[:,:,None]).tolist() for x in random];rng=np.random.default_rng(cfg['bootstrap_seed']);br=[];bg=[]
 for _ in range(cfg['bootstrap_samples']):
  ix=rng.integers(0,n,n);p=[];q=[];b=[]
  for g in groups:
   pp=pv[:,g];qq=rv[:,g];bb=bv[:,g];draw=rng.integers(0,pp.shape[1],pp.shape[1]);p.append(pp[:,draw][:,:,:,ix].mean((1,3,4)));q.append(qq[:,draw][:,:,:,ix].mean((1,3,4)));b.append(bb[:,draw][:,:,ix].mean((1,2,3)))
  p=np.stack(p,1);q=np.stack(q,1);b=np.stack(b,1);br.append(1-q/b[:,:,None]);bg.append((q-p)/b[:,:,None])
 def band(b,t):
  b=np.stack(b);rad=float(np.quantile(np.max(np.abs(b-t),axis=tuple(range(1,b.ndim))),.95));return np.stack([t-rad,t+rad],-1)
 rci=band(br,rr);gci=band(bg,gap);gates={'probability_advantage':bool((gci[:,:,:,0]>0).all()),'probability_advantage10':bool((gci[:,:,:,0]>.1).all()),'random64_half':bool((rci[:,:,1,0]>.5).all())};j={'config':cfg,'families':['actual','moment','donor'],'probability_recovery':pr.tolist(),'mean_random_recovery':rr.tolist(),'individual_random_recovery':individual,'mean_random_joint18_ci95':rci.tolist(),'probability_minus_random':gap.tolist(),'gap_joint18_ci95':gci.tolist(),'audits':audits,'gates':gates};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');np.savez_compressed(STEM.with_name(STEM.name+'_bootstrap.npz'),random=br,gap=bg);print('FINISHED',gates,'random',rr,'gap',gap,flush=True)
if __name__=='__main__':main()

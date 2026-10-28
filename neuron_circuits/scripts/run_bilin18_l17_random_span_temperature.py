"""H379 controlled output temperature diagnostic."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,B
STEM=ART/'bilin18_l17_random_span_temperature'
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
 results=[];baselines=[];audits=[];entropies=[]
 for temperature in cfg['temperatures']:
  result,base,audit,entropy=run_one(cfg,cfg['random_seeds'][0],temperature,model,pre,state,baseline,W);results.append(result);baselines.append(base);audits.append(audit);entropies.append(entropy);print('temperature',temperature,'complete',flush=True)
 analyze(cfg,np.stack(results),np.stack(baselines),audits,np.stack(entropies))
@torch.inference_mode()
def run_one(cfg,seed,temperature,model,pre,state,baseline,W):
 out=STEM.with_name(STEM.name+f'_t{temperature:g}');entropy=np.empty(len(pre))
 gen=torch.Generator(device='cpu').manual_seed(seed);q,rr=torch.linalg.qr(torch.randn(1152,64,generator=gen,dtype=torch.float64));q=q*torch.sign(torch.diag(rr))[None];V=q.to('cuda');mapped=V.T@W.T;S=len(pre);deltas=[]
 np.savez_compressed(out.with_name(out.name+f'_seed{seed}_basis.npz'),basis=q.numpy())
 for h in cfg['heads']:
  responses=np.load(ART/f'bilin18_l17_shared_geometry_precision_h{h}_responses.npz')['responses'];deltas.extend([(F.rms_norm(state+torch.tensor(v,device='cuda'),(1152,))-baseline).double() for v in responses])
 deltas=torch.stack(deltas);kl=np.empty((39,5,S));local=np.empty_like(kl);coefficients=np.zeros((39,2,S,64));normratio=np.empty((39,2,S));minrank=[16,64];stationarity=0.;nonoptimal=0.
 grid=cfg['step_grid'];gridkl=np.empty((39,2,len(grid),S))
 for off in range(0,S,8):
  sl=slice(off,off+8);d=deltas[:,sl].transpose(0,1);raw=baseline[sl].double()@W.T;dlog=d@W.T;bp=(30*torch.tanh(raw/30)/temperature).softmax(-1);cap=(1-torch.tanh(raw/30).square())/temperature;J=mapped[None]*cap[:,None];J=J-(J*bp[:,None]).sum(-1,keepdim=True);G=(J*bp[:,None])@J.transpose(1,2);targetj=dlog*cap[:,None];targetj=targetj-(targetj*bp[:,None]).sum(-1,keepdim=True);rhs=(targetj*bp[:,None])@J.transpose(1,2);euclidean=d@V;oracle=[]
  entropy[sl]=(-(bp*bp.clamp_min(1e-300).log()).sum(-1)).cpu().numpy()
  for ri,rank in enumerate(cfg['ranks']):
   g=G[:,:rank,:rank];val,vec=torch.linalg.eigh((g+g.transpose(1,2))/2);keep=val>val[:,-1:]*cfg['pinv_rtol'];minrank[ri]=min(minrank[ri],int(keep.sum(-1).min()));inv=torch.where(keep,1/val.clamp_min(1e-300),0);b=rhs[:,:,:rank];coef=((b@vec)*inv[:,None])@vec.transpose(1,2);oracle.append(coef);res=(coef@g-b).norm(dim=-1)/b.norm(dim=-1).clamp_min(1e-30);stationarity=max(stationarity,float(res.max()));coefficients[:,ri,sl,:rank]=coef.transpose(0,1).cpu().numpy();normratio[:,ri,sl]=(coef.square().sum(-1)/d.square().sum(-1)).T.cpu().numpy()
  changes=[torch.zeros_like(dlog),euclidean[:,:,:16]@mapped[:16],euclidean@mapped,oracle[0]@mapped[:16],oracle[1]@mapped];targetlp=(30*torch.tanh((raw[:,None]+dlog)/30)/temperature).log_softmax(-1);tp=targetlp.exp()
  for pi,change in enumerate(changes):
   pred=(30*torch.tanh((raw[:,None]+change)/30)/temperature).log_softmax(-1);kl[:,pi,sl]=(tp*(targetlp-pred)).sum(-1).T.cpu().numpy();err=(dlog-change)*cap[:,None];err-= (err*bp[:,None]).sum(-1,keepdim=True);local[:,pi,sl]=(bp[:,None]*err.square()).sum(-1).T.cpu().numpy()
  for ri,change in enumerate(changes[3:]):
   for gi,t in enumerate(grid):
    pred=(30*torch.tanh((raw[:,None]+t*change)/30)/temperature).log_softmax(-1);gridkl[:,ri,gi,sl]=(tp*(targetlp-pred)).sum(-1).T.cpu().numpy()
  nonoptimal=max(nonoptimal,float(np.max((local[:,3:5,sl]-local[:,1:3,sl])/local[:,:1,sl])))
  if off%256==0:print('sites',off+8,'of',S,flush=True)
 kl=kl.reshape(3,13,5,S);local=local.reshape(3,13,5,S);gridkl=gridkl.reshape(3,13,2,len(grid),S);best=gridkl.min(3)
 replay=None
 if temperature==1.:
  old=np.load(ART/f'bilin18_l17_random_span_specificity_seed{seed}_rows.npz');replay=max(float(np.max(np.abs(best-old['best_kl']))/np.max(old['best_kl'])),float(np.max(np.abs(local-old['local_error']))/np.max(old['local_error'])));assert replay<1e-8
 assert np.isfinite(gridkl).all() and gridkl.min()>-1e-10 and nonoptimal<1e-6 and stationarity<1e-6

 assert np.max(best-kl[:,:,:1])<1e-10 and np.max(best-kl[:,:,[3,4]])<1e-10
 np.savez_compressed(out.with_name(out.name+f'_seed{seed}_rows.npz'),kl=kl,local_error=local,grid_kl=gridkl,best_kl=best,chosen_index=gridkl.argmin(3),oracle_coefficients=coefficients,entropy=entropy)
 return best,kl[:,:,0],{'temperature':temperature,'seed':seed,'baseline_replay_relative':replay,'max_local_suboptimality':nonoptimal,'normal_equation_residual':stationarity,'minimum_retained_ranks':minrank,'norm_ratio_quantiles':np.quantile(np.sqrt(normratio),[.5,.95,.99,1]).tolist()},entropy

def analyze(cfg,errors,baseline,audits,entropy):
 n=cfg['rows'];groups=[slice(0,1),slice(1,7),slice(7,13)];ev=errors.reshape(3,3,13,2,n,4);bv=baseline.reshape(3,3,13,n,4)
 def compute(e,b):
  return np.stack([1-e[:,:,g].mean((2,4,5))/b[:,:,g].mean((2,3,4))[:,:,None] for g in groups],2)
 rec=compute(ev,bv);gap=rec[0]-rec[2];rng=np.random.default_rng(cfg['bootstrap_seed']);br=[];bg=[]
 for _ in range(cfg['bootstrap_samples']):
  ix=rng.integers(0,n,n);rs=[]
  for g in groups:
   e=ev[:,:,g];b=bv[:,:,g];draw=rng.integers(0,e.shape[2],e.shape[2]);em=e[:,:,draw][:,:,:,:,ix].mean((2,4,5));bm=b[:,:,draw][:,:,:,ix].mean((2,3,4));rs.append(1-em/bm[:,:,None])
  rs=np.stack(rs,2);br.append(rs);bg.append(rs[0]-rs[2])
 def band(b,t):
  b=np.stack(b);rad=float(np.quantile(np.max(np.abs(b-t),axis=tuple(range(1,b.ndim))),.95));return np.stack([t-rad,t+rad],-1)
 rci=band(br,rec);gci=band(bg,gap);entropyorder=bool((np.diff(entropy,axis=0)>=-1e-12).all());gates={'cold_hot_advantage':bool((gci[:,:,:,0]>.15).all()),'temperature_monotone':bool((np.diff(rec,axis=0)<=0).all()),'entropy_order':entropyorder};j={'config':cfg,'families':['actual','moment','donor'],'recovery':rec.tolist(),'recovery_joint54_ci95':rci.tolist(),'cold_minus_hot':gap.tolist(),'cold_minus_hot_joint18_ci95':gci.tolist(),'mean_entropy':entropy.mean(1).tolist(),'entropy_quantiles':np.quantile(entropy,[.1,.5,.9],axis=1).T.tolist(),'audits':audits,'gates':gates};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');np.savez_compressed(STEM.with_name(STEM.name+'_bootstrap.npz'),recovery=br,gap=bg);print('FINISHED',gates,'recovery',rec,'entropy',entropy.mean(1),flush=True)
if __name__=='__main__':main()

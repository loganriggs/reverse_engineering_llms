"""H380 cross temperature used to fit and score."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,B
STEM=ART/'bilin18_l17_temperature_fit_score'
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
 seed=cfg['random_seeds'][0];V=torch.tensor(np.load(ART/f'bilin18_l17_random_span_specificity_seed{seed}_basis.npz')['basis'],device='cuda');mapped=V.T@W.T;S=len(pre);deltas=[]
 for h in cfg['heads']:
  responses=np.load(ART/f'bilin18_l17_shared_geometry_precision_h{h}_responses.npz')['responses'];deltas.extend([(F.rms_norm(state+torch.tensor(v,device='cuda'),(1152,))-baseline).double() for v in responses])
 deltas=torch.stack(deltas);coeff=[];diagonal=[]
 for t in cfg['temperatures']:
  old=np.load(ART/f'bilin18_l17_random_span_temperature_t{t:g}_seed{seed}_rows.npz');steps=np.array(cfg['step_grid'])[old['chosen_index']].reshape(39,2,S);coeff.append(torch.tensor(old['oracle_coefficients']*steps[:,:,:,None],device='cuda'));diagonal.append(old['best_kl'])
 kl=np.empty((3,3,39,2,S));bk=np.empty((3,39,S))
 for off in range(0,S,8):
  sl=slice(off,off+8);raw=baseline[sl].double()@W.T;d=deltas[:,sl].transpose(0,1);targetlogits=30*torch.tanh((raw[:,None]+d@W.T)/30);baselogits=30*torch.tanh(raw/30);targets=[(targetlogits/t).log_softmax(-1) for t in cfg['temperatures']]
  for ti,t in enumerate(cfg['temperatures']):bk[ti,:,sl]=(targets[ti].exp()*(targets[ti]-(baselogits/t).log_softmax(-1)[:,None])).sum(-1).T.cpu().numpy()
  for fi,cc in enumerate(coeff):
   for ri,rank in enumerate(cfg['ranks']):
    change=cc[:,ri,sl,:rank].transpose(0,1)@mapped[:rank];logits=30*torch.tanh((raw[:,None]+change)/30)
    for ti,t in enumerate(cfg['temperatures']):
     lp=(logits/t).log_softmax(-1);kl[fi,ti,:,ri,sl]=(targets[ti].exp()*(targets[ti]-lp)).sum(-1).T.cpu().numpy()
  if off%512==0:print('sites',off+8,'of',S,flush=True)
 kl=kl.reshape(3,3,3,13,2,S);bk=bk.reshape(3,3,13,S);replay=max(float(np.max(np.abs(kl[i,i]-diagonal[i]))/np.max(diagonal[i])) for i in range(3));assert replay<1e-8 and np.isfinite(kl).all() and kl.min()>-1e-10
 np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),kl=kl,baseline_kl=bk)
 analyze(cfg,kl,bk,{'h379_diagonal_replay_relative':replay})
def analyze(cfg,kl,base,audit):
 n=cfg['rows'];groups=[slice(0,1),slice(1,7),slice(7,13)];ev=kl.reshape(3,3,3,13,2,n,4);bv=base.reshape(3,3,13,n,4)
 def stats(rec):
  total=rec[0,0]-rec[2,2];fit=.5*((rec[0,0]-rec[2,0])+(rec[0,2]-rec[2,2]));score=.5*((rec[0,0]-rec[0,2])+(rec[2,0]-rec[2,2]));interaction=rec[0,0]-rec[0,2]-rec[2,0]+rec[2,2];return np.stack([fit,score])/total[None],rec[1,1]-rec[0,1],fit,score,interaction
 rec=np.stack([1-ev[:,:,:,g].mean((3,5,6))/bv[:,:,g].mean((2,3,4))[None,:,:,None] for g in groups],3);shares,adv,fit,score,interaction=stats(rec);rng=np.random.default_rng(cfg['bootstrap_seed']);bs=[];ba=[]
 for _ in range(cfg['bootstrap_samples']):
  ix=rng.integers(0,n,n);rr=[]
  for g in groups:
   e=ev[:,:,:,g];b=bv[:,:,g];draw=rng.integers(0,e.shape[3],e.shape[3]);em=e[:,:,:,draw][:,:,:,:,:,ix].mean((3,5,6));bm=b[:,:,draw][:,:,:,ix].mean((2,3,4));rr.append(1-em/bm[None,:,:,None])
  rr=np.stack(rr,3);ss,aa,*_=stats(rr);bs.append(ss);ba.append(aa)
 def band(b,t):
  b=np.stack(b);rad=float(np.quantile(np.max(np.abs(b-t),axis=tuple(range(1,b.ndim))),.95));return np.stack([t-rad,t+rad],-1)
 sci=band(bs,shares);aci=band(ba,adv);gates={'scoring_majority':bool((sci[1,:,:,:,0]>.5).all()),'fitting_majority':bool((sci[0,:,:,:,0]>.5).all()),'native_fit_advantage':bool((aci[:,:,:,0]>.1).all())};assert np.max(np.abs(fit+score-(rec[0,0]-rec[2,2])))<1e-12
 j={'config':cfg,'families':['actual','moment','donor'],'recovery_fit_score_head_family_rank':rec.tolist(),'fit_score_shares':shares.tolist(),'shares_joint36_ci95':sci.tolist(),'fit_contribution':fit.tolist(),'score_contribution':score.tolist(),'interaction':interaction.tolist(),'native_fit_minus_cold_at_native_score':adv.tolist(),'native_advantage_joint18_ci95':aci.tolist(),'audit':audit,'gates':gates};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');np.savez_compressed(STEM.with_name(STEM.name+'_bootstrap.npz'),shares=bs,native_advantage=ba);print('FINISHED',gates,'shares',shares,'native_advantage',adv,flush=True)
if __name__=='__main__':main()

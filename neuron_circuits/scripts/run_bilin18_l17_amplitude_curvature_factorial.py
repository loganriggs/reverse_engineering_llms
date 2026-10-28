"""H391 amplitude-by-curvature factorial."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l8_execution_alignment import angle_matched_payload
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,B
def drms(v,d):
 scale=(v.square().mean(-1,keepdim=True)+torch.finfo(v.dtype).eps).sqrt();return d/scale-v*(v*d).mean(-1,keepdim=True)/scale.pow(3)
STEM=ART/'bilin18_l17_amplitude_curvature_factorial'
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
 N=cfg['rows'];amps=cfg['amplitudes'];kl=np.empty((3,13,len(amps),8,S));base=np.empty((3,13,len(amps),S));ratio=np.empty((3,13,S));identity=0.;unitreplay=0.;linearreplay=0.;native0=B(x,x,L,R,D)
 moments=np.load(ART/'bilin18_l17_source_moment_controls_moments.npz');means=[torch.tensor(v,device='cuda') for v in moments['means']];roots=[torch.tensor(v,device='cuda') for v in moments['roots']];sources=[torch.linalg.qr(model.blocks[17].attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q for h in range(9)]
 for hi,h in enumerate(cfg['heads']):
  cached=np.load(ART/f'bilin18_l17_linear_response_confirmation_h{h}_predictions.npz');variants=[('actual',None)]+[(f'moment{i}',seed) for i,seed in enumerate(cfg['draw_seeds'])]+[(f'donor{i}',shift) for i,shift in enumerate(cfg['donor_shifts'])]
  for ci,(name,seed) in enumerate(variants):
   ref=heads[:,h]
   if seed is None:pay=ref
   elif name.startswith('moment'):
    gen=torch.Generator(device='cuda').manual_seed(seed);noise=torch.randn(9,N,4,128,generator=gen,device='cuda',dtype=torch.float64);direction=torch.einsum('npi,pji->npj',noise[h],roots[h])+means[h];pay=angle_matched_payload(pre,ref,sources[h],direction.reshape(S,128).float())
   else:
    site=(((torch.arange(N,device='cuda')+seed)%N)[:,None]*4+torch.arange(4,device='cuda')).reshape(-1);pay=angle_matched_payload(pre,ref,sources[h],heads[site,h]@sources[h])
   ratio[hi,ci]=(pay.norm(dim=-1)/pre.norm(dim=-1)).cpu().numpy();dx=drms(pre,-pay)
   for ai,amp in enumerate(amps):
    exactx=F.rms_norm(pre-amp*pay,(1152,));native=B(exactx,exactx,L,R,D)-native0;targetdelta=(F.rms_norm(state+native,(1152,))-baseline).double();preds=[]
    for inputexact in [False,True]:
     u=exactx-x if inputexact else amp*dx;cross=B(x,u,L,R,D)+B(u,x,L,R,D);quadratic=B(u,u,L,R,D)
     for includeq in [False,True]:
      m=cross+quadratic if includeq else cross
      for outputexact in [False,True]:preds.append(F.rms_norm(state+m,(1152,))-baseline if outputexact else drms(state,m))
    preds=torch.stack(preds).double()
    if amp==1.:
     unitreplay=max(unitreplay,float((native-torch.tensor(cached['native_responses'][ci],device='cuda')).abs().max()));linearreplay=max(linearreplay,float((preds[0]-torch.tensor(cached['linear_post_rms'][ci],device='cuda')).abs().max()))
    for off in range(0,S,16):
     sl=slice(off,off+16);raw=baseline[sl].double()@W.T;targetlp=(30*torch.tanh((raw+targetdelta[sl]@W.T)/30)).log_softmax(-1);tp=targetlp.exp();predlp=(30*torch.tanh((raw[None]+preds[:,sl]@W.T)/30)).log_softmax(-1);kl[hi,ci,ai,:,sl]=(tp[None]*(targetlp[None]-predlp)).sum(-1).cpu().numpy();base[hi,ci,ai,sl]=(tp*(targetlp-(30*torch.tanh(raw/30)).log_softmax(-1))).sum(-1).cpu().numpy()
   if ci%3==0:print('head',h,'condition',ci,'of12 done',flush=True)
 identity=float(np.max(np.abs(kl[:,:,:,7])));assert np.isfinite(kl).all() and np.isfinite(base).all() and kl.min()>-1e-10 and base.min()>0 and identity<1e-9 and unitreplay<1e-7 and linearreplay<1e-7
 np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),kl=kl,baseline_kl=base,payload_over_pre_norm=ratio);analyze(cfg,kl,base,{'all_exact_corner_max_kl':identity,'unit_native_response_max_absolute':unitreplay,'unit_linear_prediction_max_absolute':linearreplay,'payload_over_pre_norm_quantiles':np.quantile(ratio,[.5,.95,.99,1]).tolist()})
def analyze(cfg,kl,base,audit):
 n=cfg['rows'];groups=[slice(0,1),slice(1,7),slice(7,13)];errors=kl.reshape(3,13,4,8,n,4);base=base.reshape(3,13,4,n,4)
 def shapley(rec):
  vals=[]
  for bit in [4,2,1]:
   v=np.zeros(rec.shape[:-1])
   for mask in range(8):
    if mask&bit:continue
    size=mask.bit_count();weight=1/3 if size in [0,2] else 1/6;v+=weight*(rec[...,mask|bit]-rec[...,mask])
   vals.append(v)
  return np.stack(vals,-1)
 rec=np.stack([1-errors[:,g].mean((1,4,5))/base[:,g].mean((1,3,4))[:,:,None] for g in groups],1);contrib=shapley(rec);total=rec[...,7]-rec[...,0];assert (total>0).all();shares=contrib/total[...,None];rng=np.random.default_rng(cfg['bootstrap_seed']);br=[];bs=[]
 for _ in range(cfg['bootstrap_samples']):
  ix=rng.integers(0,n,n);rr=[]
  for g in groups:
   e=errors[:,g];b=base[:,g];draw=rng.integers(0,e.shape[1],e.shape[1]);em=e[:,draw][:,:,:,:,ix].mean((1,4,5));bm=b[:,draw][:,:,:,ix].mean((1,3,4));rr.append(1-em/bm[:,:,None])
  rr=np.stack(rr,1);br.append(rr);bs.append(shapley(rr)/(rr[...,7]-rr[...,0])[...,None])
 br,bs=map(np.stack,[br,bs]);rads=np.quantile(np.abs(br-rec).max(axis=(1,2,4)),.95,axis=0);srads=np.quantile(np.abs(bs-shares).max(axis=(1,2,4)),.95,axis=0);rci=np.stack([rec-rads[None,None,:,None],rec+rads[None,None,:,None]],-1);sci=np.stack([shares-srads[None,None,:,None],shares+srads[None,None,:,None]],-1);gates={'linear_unit90':bool((rci[:,:,1,0,0]>.9).all()),'linear_four90':bool((rci[:,:,3,0,0]>.9).all()),'input_majority_four':bool((sci[:,:,3,0,0]>.5).all()),'quadratic_majority_four':bool((sci[:,:,3,1,0]>.5).all()),'output_majority_four':bool((sci[:,:,3,2,0]>.5).all())};j={'config':cfg,'families':['actual','moment','donor'],'bit_order':{'input_exact':4,'quadratic_included':2,'output_exact':1},'recovery':rec.tolist(),'recovery_joint72_per_amplitude_ci95':rci.tolist(),'shapley_contributions':contrib.tolist(),'shapley_shares':shares.tolist(),'shares_joint27_per_amplitude_ci95':sci.tolist(),'audit':audit,'gates':gates};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');np.savez_compressed(STEM.with_name(STEM.name+'_bootstrap.npz'),recovery=br,shares=bs);print('FINISHED',gates,'linear recovery',rec[...,0],'shares4',shares[:,:,3],flush=True)
if __name__=='__main__':main()

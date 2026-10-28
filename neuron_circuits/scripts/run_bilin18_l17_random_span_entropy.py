"""H378 entropy-stratified oracle recovery."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,B
STEM=ART/'bilin18_l17_random_span_entropy'
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
 metrics=[]
 for off in range(0,len(pre),16):
  lp=(30*torch.tanh((baseline[off:off+16].double()@W.T)/30)).log_softmax(-1);p=lp.exp();metrics.append(torch.stack([-(p*lp).sum(-1),1/p.square().sum(-1),p.max(-1).values],-1).cpu().numpy())
 metrics=np.concatenate(metrics);assert np.isfinite(metrics).all();bins=np.digitize(metrics[:,0],cfg['entropy_edges'][1:-1]);counts=np.bincount(bins,minlength=4);doccounts=[int((bins.reshape(512,4)==i).any(1).sum()) for i in range(4)];np.savez_compressed(STEM.with_name(STEM.name+'_features.npz'),entropy=metrics[:,0],inverse_probability_second_moment=metrics[:,1],max_probability=metrics[:,2],bins=bins)
 old=np.load(ART/'bilin18_l17_probability_axes_oracle_rows.npz');baselinekl=old['kl'][:,:,0];prob=np.load(ART/'bilin18_l17_oracle_finite_safeguard_rows.npz')['best_kl'];random=np.mean([np.load(ART/f'bilin18_l17_random_span_specificity_seed{seed}_rows.npz')['best_kl'] for seed in cfg['random_seeds']],axis=0);analyze(cfg,bins,counts,doccounts,baselinekl,prob,random)
def analyze(cfg,bins,counts,doccounts,baselinekl,prob,random):
 n=cfg['rows'];groups=[slice(0,1),slice(1,7),slice(7,13)];base=baselinekl.reshape(3,13,n,4);errors=np.stack([prob,random],2).reshape(3,13,2,2,n,4);mask=np.stack([bins.reshape(n,4)==i for i in range(4)])
 # Sum each document's errors and baseline over its sites in each entropy bin.
 eb=np.einsum('hcarkp,bkp->hcarkb',errors,mask);bb=np.einsum('hckp,bkp->hckb',base,mask)
 def compute(ee,be):
  r=[]
  for g in groups:
   x=ee[:,g].mean(1).sum(-2);b=be[:,g].mean(1).sum(-2);r.append(1-x/b[:,None,None,:])
  return np.stack(r,1)
 rec=compute(eb,bb);contrast=rec[:,:,1,1,0]-rec[:,:,1,1,3];rng=np.random.default_rng(cfg['bootstrap_seed']);bc=[];br=[]
 for _ in range(cfg['bootstrap_samples']):
  ix=rng.integers(0,n,n);rr=[]
  for g in groups:
   e=eb[:,g];b=bb[:,g];draw=rng.integers(0,e.shape[1],e.shape[1]);ev=e[:,draw][:,:,:,:,ix].mean(1).sum(-2);bv=b[:,draw][:,:,ix].mean(1).sum(-2);rr.append(1-ev/bv[:,None,None,:])
  rr=np.stack(rr,1);br.append(rr[:,:,1,1]);bc.append(rr[:,:,1,1,0]-rr[:,:,1,1,3])
 bc=np.stack(bc);br=np.stack(br);assert np.isfinite(br).all();cr=float(np.quantile(np.max(np.abs(bc-contrast),axis=(1,2)),.95));rad=float(np.quantile(np.max(np.abs(br-rec[:,:,1,1]),axis=(1,2,3)),.95));ci=np.stack([contrast-cr,contrast+cr],-1);gates={'adequate_extremes':doccounts[0]>=50 and doccounts[3]>=50,'random64_low_high':bool((ci[:,:,0]>.15).all()),'random64_monotone':bool((np.diff(rec[:,:,1,1],axis=-1)<=0).all())}
 j={'config':cfg,'families':['actual','moment','donor'],'basis_order':['probability','mean_random'],'rank_order':[16,64],'site_counts':counts.tolist(),'distinct_document_counts':doccounts,'recovery':rec.tolist(),'random64_joint36_ci95':np.stack([rec[:,:,1,1]-rad,rec[:,:,1,1]+rad],-1).tolist(),'random64_low_minus_high':contrast.tolist(),'random64_contrast_joint9_ci95':ci.tolist(),'gates':gates};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');np.savez_compressed(STEM.with_name(STEM.name+'_bootstrap.npz'),random64=br,contrast=bc);print('FINISHED',gates,'counts',counts,doccounts,'random64',rec[:,:,1,1],'contrast',contrast,flush=True)
if __name__=='__main__':main()

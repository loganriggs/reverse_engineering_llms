import json,hashlib,numpy as np,torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources
STEM=ART/'bilin18_l17_metric_sample_confirmation'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,digest in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==digest
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();docs=np.load(ROOT/cfg['corpus']);p,dh,split=collect_sources(model,docs,cfg);dh=dh.double();assert split<1e-5 and cfg['positions']==[63] and cfg['start']==0;targets=torch.tensor(docs[:,64].astype(np.int64),device='cuda');z=np.load(ART/'bilin18_l17_metric_sample_curve_models.npz');B=torch.tensor(z['basis'],device='cuda');mu=torch.tensor(z['mean'],device='cuda');cs=torch.tensor(z['covariance'],device='cuda').sqrt();h=dh.sum(1)-mu;hc=h@B;errors={'mean':h}
 for k in cfg['ranks']:
  errors[f'PCA{k}']=h-hc[:,:k]@B[:,:k].T
  for n in [64,1024]:
   u=torch.tensor(z[f'U{n}'][:,:k],device='cuda');errors[f'n{n}_rank{k}']=h-(((hc/cs)@u)@u.T*cs)@B.T
 L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];bias=model.blocks[17].mlp.Down_bias.detach();W=model.lm_head.weight.detach();saved={};metrics={}
 def native(x):
  v=F.rms_norm(x,(1152,));return ((v@L.T)*(v@R.T))@D.T+bias
 def lp(x):return (30*torch.tanh(F.rms_norm(x,(1152,))@W.T/30)).double().log_softmax(-1)
 y0=native(p)
 for name,e in errors.items():
  inp=p-e.float();yp=native(inp);rows=[]
  for start in range(0,512,16):
   sl=slice(start,start+16);base=lp(p[sl]+y0[sl]);q=lp(inp[sl]+yp[sl]);rows.append(torch.stack([(base.exp()*(base-q)).sum(-1),(base-q).gather(1,targets[sl,None]).squeeze(1)],-1).cpu().numpy())
  rr=np.concatenate(rows);saved[name]=rr;metrics[name]={'mean_KL':float(rr[:,0].mean()),'mean_CE_change':float(rr[:,1].mean()),'CE_change_SE':float(rr[:,1].std(ddof=1)/np.sqrt(512)),'source_raw_recovery':float(1-e.square().sum()/h.square().sum())};print(name,metrics[name],flush=True)
 for n in metrics:metrics[n]['KL_recovery_vs_mean']=float(1-metrics[n]['mean_KL']/metrics['mean']['mean_KL'])
 comparisons=[];dd=[]
 for k in cfg['ranks']:
  for comparator in [f'n64_rank{k}',f'PCA{k}']:
   comparisons.append(f'n1024_rank{k}_over_{comparator}');dd.append(saved[comparator][:,0]-saved[f'n1024_rank{k}'][:,0])
 diffs=np.stack(dd,1);est=diffs.mean(0);sd=diffs.std(0,ddof=1)/np.sqrt(512);assert (sd>0).all();rng=np.random.default_rng(cfg['bootstrap_seed']);idx=rng.integers(0,512,(cfg['bootstrap_draws'],512));draw=diffs[idx].mean(1);q=float(np.quantile(np.max(np.abs((draw-est)/sd),axis=1),.95))
 joint={'comparisons':comparisons,'mean_KL_reductions':est.tolist(),'paired_SE':sd.tolist(),'lower':(est-q*sd).tolist(),'upper':(est+q*sd).tolist(),'positive_lower_gates':(est-q*sd>0).tolist(),'critical_value':q,'method':'centered max absolute standardized document bootstrap; fixed original paired SEs; simultaneous two-sided95% family of6'}
 assert all(np.isfinite(v).all() for v in saved.values());np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**saved,pre=p.cpu().numpy(),head_writes=dh.cpu().numpy(),targets=targets.cpu().numpy(),paired_KL_gains=diffs)
 STEM.with_suffix('.json').write_text(json.dumps({'experiment':'H521','config':cfg,'metrics':metrics,'joint_gains':joint,'head_write_sum_relative':split,'unique_documents':22016},indent=2)+'\n');print('joint',joint,flush=True)
if __name__=='__main__':main()

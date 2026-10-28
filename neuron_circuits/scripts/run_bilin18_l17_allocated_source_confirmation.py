import json,hashlib,numpy as np,torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources
STEM=ART/'bilin18_l17_allocated_source_confirmation'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,h in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==h
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();docs=np.load(ROOT/cfg['corpus']);p,dh,spliterror=collect_sources(model,docs,cfg);dh=dh.double();targets=torch.tensor(docs[:,64].astype(np.int64),device='cuda');assert cfg['positions']==[63] and cfg['start']==0 and len(docs)==512 and spliterror<1e-5;compiled=np.load(ART/'bilin18_l17_fisher_source_rank_compiled.npz');pooled=np.load(ART/'bilin18_l17_fisher_pooled_metric_models.npz');L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];bias=model.blocks[17].mlp.Down_bias.detach();W=model.lm_head.weight.detach();saved={};metrics={};errs={n:torch.zeros_like(p,dtype=torch.float64) for n in cfg['methods']};total=torch.zeros_like(p,dtype=torch.float64)
 def native(x):
  v=F.rms_norm(x,(1152,));return ((v@L.T)*(v@R.T))@D.T+bias
 def lp(x):return (30*torch.tanh(F.rms_norm(x,(1152,))@W.T/30)).double().log_softmax(-1)
 y0=native(p);full=np.load(ART/'bilin18_l17_full_path_metric_models.npz');allocated=np.load(ART/'bilin18_l17_allocated_source_rank_models.npz');allocation=json.loads((ART/'bilin18_l17_allocated_source_rank_allocation.json').read_text());old=np.load(ART/'bilin18_l17_full_path_metric_rows.npz')
 for head in cfg['heads']:
  B=torch.tensor(compiled[f'h{head}_basis'],device='cuda');cs=torch.tensor(compiled[f'h{head}_covariance'],device='cuda').sqrt();U=torch.tensor(compiled[f'h{head}_weighted_eigenvectors'][:,:4],device='cuda');A=torch.tensor(pooled[f'h{head}_A'],device='cuda');h=dh[:,head]-torch.tensor(compiled[f'h{head}_mean'],device='cuda');Q=torch.linalg.qr(B@(cs[:,None]*U)).Q;hc=h@B;maps={'mean':torch.zeros_like(h),'PCA':hc[:,:4]@B[:,:4].T,'orthogonal':(h@Q)@Q.T,'original':(((hc/cs)@U)@U.T*cs)@B.T,'pooled':hc@A.T@B.T,'full_metric':hc@torch.tensor(full[f'h{head}_A'],device='cuda').T@B.T,'allocated':hc@torch.tensor(allocated[f'h{head}_A'],device='cuda').T@B.T};total+=h
  for n in errs:errs[n]+=h-maps[n]
 for path in cfg['paths']:
  metrics[path]={}
  for name,error in errs.items():
   inp=p-error.float();yp=native(inp);bypass=p if path=='MLP_only' else inp;rows=[]
   for start in range(0,512,16):
    sl=slice(start,start+16);base=lp(p[sl]+y0[sl]);q=lp(bypass[sl]+yp[sl]);rows.append(torch.stack([(base.exp()*(base-q)).sum(-1),(base-q).gather(1,targets[sl,None]).squeeze(1),(bypass[sl].double()+yp[sl].double()-p[sl].double()-y0[sl].double()).square().sum(-1)],-1).cpu().numpy())
   rr=np.concatenate(rows);
   saved[f'{path}_{name}']=rr;metrics[path][name]={'mean_KL':float(rr[:,0].mean()),'mean_CE_change':float(rr[:,1].mean()),'CE_change_SE':float(rr[:,1].std(ddof=1)/np.sqrt(512)),'mean_state_squared_error':float(rr[:,2].mean()),'source_raw_recovery':float(1-error.square().sum()/total.square().sum())}
  for name in errs:metrics[path][name]['KL_recovery_vs_mean']=float(1-metrics[path][name]['mean_KL']/metrics[path]['mean']['mean_KL'])
  print(path,metrics[path],flush=True)
 assert all(np.isfinite(v).all() for v in saved.values());names=['full_metric','PCA'];diffs=np.stack([saved['full_pre_MLP_'+n][:,0]-saved['full_pre_MLP_allocated'][:,0] for n in names],1);est=diffs.mean(0);rng=np.random.default_rng(cfg['bootstrap_seed']);idx=rng.integers(0,512,(cfg['bootstrap_draws'],512));draw=diffs[idx].mean(1);sd=draw.std(0,ddof=1);assert (sd>0).all();critical=float(np.quantile(np.max(np.abs((draw-est)/sd),axis=1),.95));joint={'comparators':names,'mean_KL_reductions':est.tolist(),'lower':(est-critical*sd).tolist(),'upper':(est+critical*sd).tolist(),'positive_lower_gates':(est-critical*sd>0).tolist(),'critical_value':critical,'scope':'simultaneous95% paired iid document bootstrap; frozen models/calibration/training'};
 np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**saved,pre=p.cpu().numpy(),head_writes=dh.cpu().numpy(),targets=targets.cpu().numpy());STEM.with_suffix('.json').write_text(json.dumps({'experiment':'H510','config':cfg,'metrics':metrics,'joint_gains':joint,'head_write_sum_relative':spliterror,'allocation':allocation,'columns':['KL_original_to_replacement','observed_token_CE_change','final_state_squared_error'],'unique_documents':19968},indent=2)+'\n')
if __name__=='__main__':main()

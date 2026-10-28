import json,hashlib,numpy as np,torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model
STEM=ART/'bilin18_l17_joint_source_paths'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,h in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==h
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();data=np.load(ART/'bilin18_l17_natural_source_replacement_rows.npz');p=torch.tensor(data['pre'],device='cuda');dh=torch.tensor(data['head_writes'],device='cuda').double();targets=torch.tensor(data['targets'],device='cuda');compiled=np.load(ART/'bilin18_l17_fisher_source_rank_compiled.npz');pooled=np.load(ART/'bilin18_l17_fisher_pooled_metric_models.npz');L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];bias=model.blocks[17].mlp.Down_bias.detach();W=model.lm_head.weight.detach();saved={};metrics={};errs={n:torch.zeros_like(p,dtype=torch.float64) for n in cfg['methods']};total=torch.zeros_like(p,dtype=torch.float64)
 def native(x):
  v=F.rms_norm(x,(1152,));return ((v@L.T)*(v@R.T))@D.T+bias
 def lp(x):return (30*torch.tanh(F.rms_norm(x,(1152,))@W.T/30)).double().log_softmax(-1)
 y0=native(p)
 for head in cfg['heads']:
  B=torch.tensor(compiled[f'h{head}_basis'],device='cuda');cs=torch.tensor(compiled[f'h{head}_covariance'],device='cuda').sqrt();U=torch.tensor(compiled[f'h{head}_weighted_eigenvectors'][:,:4],device='cuda');A=torch.tensor(pooled[f'h{head}_A'],device='cuda');h=dh[:,head]-torch.tensor(compiled[f'h{head}_mean'],device='cuda');Q=torch.linalg.qr(B@(cs[:,None]*U)).Q;hc=h@B;maps={'mean':torch.zeros_like(h),'PCA':hc[:,:4]@B[:,:4].T,'orthogonal':(h@Q)@Q.T,'original':(((hc/cs)@U)@U.T*cs)@B.T,'pooled':hc@A.T@B.T};total+=h
  for n in errs:errs[n]+=h-maps[n]
 for path in cfg['paths']:
  metrics[path]={}
  for name,error in errs.items():
   inp=p-error.float();yp=native(inp);bypass=p if path=='MLP_only' else inp;rows=[]
   for start in range(0,512,16):
    sl=slice(start,start+16);base=lp(p[sl]+y0[sl]);q=lp(bypass[sl]+yp[sl]);rows.append(torch.stack([(base.exp()*(base-q)).sum(-1),(base-q).gather(1,targets[sl,None]).squeeze(1),(bypass[sl].double()+yp[sl].double()-p[sl].double()-y0[sl].double()).square().sum(-1)],-1).cpu().numpy())
   rr=np.concatenate(rows);saved[f'{path}_{name}']=rr;metrics[path][name]={'mean_KL':float(rr[:,0].mean()),'mean_CE_change':float(rr[:,1].mean()),'CE_change_SE':float(rr[:,1].std(ddof=1)/np.sqrt(512)),'mean_state_squared_error':float(rr[:,2].mean()),'source_raw_recovery':float(1-error.square().sum()/total.square().sum())}
  for name in errs:metrics[path][name]['KL_recovery_vs_mean']=float(1-metrics[path][name]['mean_KL']/metrics[path]['mean']['mean_KL'])
  print(path,metrics[path],flush=True)
 assert all(np.isfinite(v).all() for v in saved.values());np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**saved);STEM.with_suffix('.json').write_text(json.dumps({'experiment':'H506','config':cfg,'metrics':metrics,'columns':['KL_original_to_replacement','observed_token_CE_change','final_state_squared_error'],'unique_documents':19456},indent=2)+'\n')
if __name__=='__main__':main()

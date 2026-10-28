import json,hashlib,numpy as np,torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources
STEM=ART/'bilin18_l17_natural_source_replacement'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,h in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==h
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();docs=np.load(ROOT/cfg['corpus']);p,dh,spliterror=collect_sources(model,docs,cfg);assert spliterror<1e-5;oldp=torch.tensor(np.load(ART/'bilin18_l17_fisher_source_confirmation_rows.npz')['pre'],device='cuda');replay=float((p-oldp).norm()/oldp.norm());assert replay<1e-6;targets=torch.tensor(docs[:,64].astype(np.int64),device='cuda');assert cfg['positions']==[63] and cfg['start']==0 and len(docs)==512;compiled=np.load(ART/'bilin18_l17_fisher_source_rank_compiled.npz');pooled=np.load(ART/'bilin18_l17_fisher_pooled_metric_models.npz');L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];bias=model.blocks[17].mlp.Down_bias.detach();W=model.lm_head.weight.detach();saved={};metrics={};sourcegeom={};audits={'head_write_sum_relative':spliterror,'pre_replay_relative':replay}
 def native(x):
  v=F.rms_norm(x,(1152,));return ((v@L.T)*(v@R.T))@D.T+bias
 def lp(x):return (30*torch.tanh(F.rms_norm(x,(1152,))@W.T/30)).double().log_softmax(-1)
 y0=native(p);baseline=[]
 for start in range(0,512,16):
  sl=slice(start,start+16);base=lp(p[sl]+y0[sl]);baseline.append(-base.gather(1,targets[sl,None]).squeeze(1).cpu().numpy())
 saved['baseline_CE']=np.concatenate(baseline)
 for head in cfg['heads']:
  B=torch.tensor(compiled[f'h{head}_basis'],device='cuda');cs=torch.tensor(compiled[f'h{head}_covariance'],device='cuda').sqrt();U=torch.tensor(compiled[f'h{head}_weighted_eigenvectors'][:,:4],device='cuda');A=torch.tensor(pooled[f'h{head}_A'],device='cuda');h=dh[:,head].double()-torch.tensor(compiled[f'h{head}_mean'],device='cuda');Q=torch.linalg.qr(B@(cs[:,None]*U)).Q;hc=h@B;maps={'mean':torch.zeros_like(h),'PCA':hc[:,:4]@B[:,:4].T,'orthogonal':(h@Q)@Q.T,'original':(((hc/cs)@U)@U.T*cs)@B.T,'pooled':hc@A.T@B.T};metrics[str(head)]={};sourcegeom[str(head)]={}
  for name,hhat in maps.items():
   error=h-hhat;yp=native(p-error.float());rows=[];sourcegeom[str(head)][name]={'source_raw_recovery':float(1-error.square().sum()/h.square().sum())}
   for start in range(0,512,16):
    sl=slice(start,start+16);base=lp(p[sl]+y0[sl]);q=lp(p[sl]+yp[sl]);rows.append(torch.stack([(base.exp()*(base-q)).sum(-1),(base-q).gather(1,targets[sl,None]).squeeze(1),(yp[sl].double()-y0[sl].double()).square().sum(-1)],-1).cpu().numpy())
   rr=np.concatenate(rows);saved[f'h{head}_{name}']=rr;metrics[str(head)][name]={'mean_KL':float(rr[:,0].mean()),'mean_CE_change':float(rr[:,1].mean()),'CE_change_SE':float(rr[:,1].std(ddof=1)/np.sqrt(len(rr))),'mean_raw_write_error':float(rr[:,2].mean())}
  for name in maps:metrics[str(head)][name]['KL_recovery_vs_mean']=float(1-metrics[str(head)][name]['mean_KL']/metrics[str(head)]['mean']['mean_KL'])
  print(head,metrics[str(head)],flush=True)
 assert all(np.isfinite(v).all() for v in saved.values());np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**saved,pre=p.cpu().numpy(),head_writes=dh.cpu().numpy(),targets=targets.cpu().numpy());STEM.with_suffix('.json').write_text(json.dumps({'experiment':'H505','config':cfg,'metrics':metrics,'source_geometry':sourcegeom,'audits':audits,'baseline_mean_CE':float(saved['baseline_CE'].mean()),'columns':['KL_original_to_replacement','observed_token_CE_change','raw_MLP_write_error'],'unique_documents':19456},indent=2)+'\n')
if __name__=='__main__':main()

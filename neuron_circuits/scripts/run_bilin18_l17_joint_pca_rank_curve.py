import json,hashlib,numpy as np,torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model
STEM=ART/'bilin18_l17_joint_pca_rank_curve'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,digest in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==digest
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();z=np.load(ART/'bilin18_l17_joint_pca_more_calibration_models.npz');B=torch.tensor(z['pooled1024_basis'],device='cuda');mu=torch.tensor(z['shared_reconstruction_mean'],device='cuda');old=np.load(ART/'bilin18_l17_joint_source_confirmation_rows.npz');p=torch.tensor(old['pre'],device='cuda');h=torch.tensor(old['head_writes'],device='cuda').double().sum(1)-mu;targets=torch.tensor(old['targets'],device='cuda');L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];bias=model.blocks[17].mlp.Down_bias.detach();W=model.lm_head.weight.detach();metrics={};saved={};den=float(old['mean'][:,0].mean())
 def native(x):
  v=F.rms_norm(x,(1152,));return ((v@L.T)*(v@R.T))@D.T+bias
 def lp(x):return (30*torch.tanh(F.rms_norm(x,(1152,))@W.T/30)).double().log_softmax(-1)
 y0=native(p)
 for k in cfg['ranks']:
  b=B[:,:k];error=h-(h@b)@b.T;inp=p-error.float();yp=native(inp);rows=[]
  for start in range(0,512,16):
   sl=slice(start,start+16);base=lp(p[sl]+y0[sl]);q=lp(inp[sl]+yp[sl]);rows.append(torch.stack([(base.exp()*(base-q)).sum(-1),(base-q).gather(1,targets[sl,None]).squeeze(1)],-1).cpu().numpy())
  rr=np.concatenate(rows);saved[f'rank{k}']=rr;metrics[str(k)]={'mean_KL':float(rr[:,0].mean()),'KL_recovery_vs_mean':float(1-rr[:,0].mean()/den),'mean_CE_change':float(rr[:,1].mean()),'CE_change_SE':float(rr[:,1].std(ddof=1)/np.sqrt(512)),'source_raw_retention':float(1-error.square().sum()/h.square().sum())};print(k,metrics[str(k)],flush=True)
  if k==36:assert np.allclose(rr,np.load(ART/'bilin18_l17_joint_pca_more_calibration_rows.npz')['rows'][:,6:8],rtol=1e-5,atol=1e-9)
 assert all(np.isfinite(v).all() for v in saved.values());np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**saved);STEM.with_suffix('.json').write_text(json.dumps({'experiment':'H515','config':cfg,'metrics':metrics,'mean_replacement_KL':den,'unique_documents':21504},indent=2)+'\n')
if __name__=='__main__':main()

import json,hashlib,numpy as np,torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model
STEM=ART/'bilin18_l17_context_delta_transfer'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,d in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==d
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float()
 def T(x):return torch.tensor(x,device='cuda',dtype=torch.float64)
 src=np.load(ART/'bilin18_l17_history_transfer_sources.npz');z=np.load(ART/'bilin18_l17_source_sample_curve_models.npz');B=T(z['basis']);p=torch.tensor(src['pre'][:,3],device='cuda');ss=T(src['sum_writes']);targets=torch.tensor(src['targets'][:,3],device='cuda');assert np.all(src['targets']==src['targets'][:,3,None])
 L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];bias=model.blocks[17].mlp.Down_bias.detach();W=model.lm_head.weight.detach()
 def native(x):
  v=F.rms_norm(x,(1152,));return ((v@L.T)*(v@R.T))@D.T+bias
 def lp(x):return (30*torch.tanh(F.rms_norm(x,(1152,))@W.T/30)).double().log_softmax(-1)
 y=native(p);names=['zero','identity']+[f'{method}{k}' for k in cfg['ranks'] for method in ['PCA','Fisher']];saved={};metrics={};audits={}
 for i,length in enumerate(cfg['short_lengths']):
  delta=ss[:,3]-ss[:,i];dc=delta@B;metrics[str(length)]={};saved[f'L{length}_delta_energy']=delta.square().sum(-1).cpu().numpy()
  for name in names:
   e=delta if name=='zero' else torch.zeros_like(delta) if name=='identity' else delta-dc@T(z['n3072_'+name]).T@B.T
   inp=p-e.float();yp=native(inp);rows=[]
   for start in range(0,2048,16):
    sl=slice(start,start+16);base=lp(p[sl]+y[sl]);q=lp(inp[sl]+yp[sl]);rows.append(torch.stack([(base.exp()*(base-q)).sum(-1),(base-q).gather(1,targets[sl,None]).squeeze(1)],-1).cpu().numpy())
   rr=np.concatenate(rows);assert np.isfinite(rr).all();saved[f'L{length}_{name}']=rr;metrics[str(length)][name]={'mean_KL':float(rr[:,0].mean()),'mean_CE_change':float(rr[:,1].mean()),'CE_change_SE':float(rr[:,1].std(ddof=1)/np.sqrt(2048)),'raw_delta_recovery':float(1-e.square().sum()/delta.square().sum())}
   if name=='identity':audits[f'L{length}_identity_max_absolute']=float(np.abs(rr).max());assert np.abs(rr).max()==0
  for name,m in metrics[str(length)].items():m['KL_recovery_vs_zero_delta']=1-m['mean_KL']/metrics[str(length)]['zero']['mean_KL']
  print(length,metrics[str(length)],flush=True)
 np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**saved);STEM.with_suffix('.json').write_text(json.dumps({'experiment':'H531','config':cfg,'metrics':metrics,'audits':audits,'unique_documents':26112},indent=2)+'\n')
if __name__=='__main__':main()

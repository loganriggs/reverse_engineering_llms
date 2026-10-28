import json,hashlib,numpy as np,torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources
STEM=ART/'bilin18_l17_position_transfer'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,d in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==d
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();docs=np.load(ROOT/cfg['corpus']);p,heads,split=collect_sources(model,docs,cfg);assert split<1e-5;ss=heads.double().sum(1);del heads
 def T(x):return torch.tensor(x,device='cuda',dtype=torch.float64)
 z=np.load(ART/'bilin18_l17_source_sample_curve_models.npz');B=T(z['basis']);mu=T(z['mean']);h=ss-mu;hc=h@B;targets=torch.tensor(docs[:,np.array(cfg['positions'])+1].astype(np.int64).reshape(-1),device='cuda')
 prior=np.load(ART/'bilin18_l17_label_direction_confirmation_rows.npz');audits={'head_sum_relative':split,'position63_pre_relative':float((p.reshape(2048,4,-1)[:,1]-torch.tensor(prior['pre'],device='cuda')).norm()/p.reshape(2048,4,-1)[:,1].norm())};assert audits['position63_pre_relative']<1e-6
 np.savez_compressed(STEM.with_name(STEM.name+'_sources.npz'),pre=p.cpu().numpy().reshape(2048,4,-1),sum_writes=ss.cpu().numpy().reshape(2048,4,-1),targets=targets.cpu().numpy().reshape(2048,4))
 L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];bias=model.blocks[17].mlp.Down_bias.detach();W=model.lm_head.weight.detach()
 def native(x):
  v=F.rms_norm(x,(1152,));return ((v@L.T)*(v@R.T))@D.T+bias
 def lp(x):return (30*torch.tanh(F.rms_norm(x,(1152,))@W.T/30)).double().log_softmax(-1)
 y=native(p);saved={};metrics={str(pos):{} for pos in cfg['positions']};names=['mean']+[f'{method}{k}' for k in cfg['ranks'] for method in ['PCA','Fisher']]
 for name in names:
  e=h if name=='mean' else h-hc@T(z['n3072_'+name]).T@B.T
  inp=p-e.float();yp=native(inp);rr=[]
  for start in range(0,len(p),16):
   sl=slice(start,start+16);base=lp(p[sl]+y[sl]);q=lp(inp[sl]+yp[sl]);rr.append(torch.stack([(base.exp()*(base-q)).sum(-1),(base-q).gather(1,targets[sl,None]).squeeze(1)],-1).cpu().numpy())
  rr=np.concatenate(rr).reshape(2048,4,2);energy=e.square().sum(-1).cpu().numpy().reshape(2048,4);sourceenergy=h.square().sum(-1).cpu().numpy().reshape(2048,4);assert np.isfinite(rr).all();saved[name]=rr;saved[name+'_raw_error_energy']=energy
  for i,pos in enumerate(cfg['positions']):
   vals=rr[:,i];metrics[str(pos)][name]={'mean_KL':float(vals[:,0].mean()),'mean_CE_change':float(vals[:,1].mean()),'CE_change_SE':float(vals[:,1].std(ddof=1)/np.sqrt(2048)),'raw_recovery':float(1-energy[:,i].sum()/sourceenergy[:,i].sum())}
  print(name,{pos:metrics[pos][name]['mean_KL'] for pos in metrics},flush=True)
 for pos,mm in metrics.items():
  for name,m in mm.items():m['KL_recovery']=1-m['mean_KL']/mm['mean']['mean_KL']
 replay=abs(saved['Fisher36'][:,1,0].mean()/prior['losses'][:,1].mean()-1);audits['position63_Fisher36_KL_relative_replay']=float(replay);assert replay<1e-4
 saved['source_energy']=sourceenergy;np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**saved);STEM.with_suffix('.json').write_text(json.dumps({'experiment':'H529','config':cfg,'metrics':metrics,'audits':audits,'unique_documents':26112},indent=2)+'\n')
if __name__=='__main__':main()

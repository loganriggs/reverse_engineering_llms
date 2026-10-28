import json,hashlib,numpy as np,torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/bilin18_tensor_decomposition';STEM=ART/'bilin18_l17_source_moment_diagnostic'
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,digest in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==digest
 z=np.load(ART/'bilin18_l17_joint_fisher_rank_curve_model.npz');calz=np.load(ART/'bilin18_l17_joint_pca_more_calibration_models.npz');evz=np.load(ART/'bilin18_l17_joint_source_confirmation_rows.npz');B=torch.tensor(z['basis'],device='cuda');mu=torch.tensor(z['mean'],device='cuda');cs=torch.tensor(z['covariance'],device='cuda').sqrt();F=torch.tensor(z['F'],device='cuda');U=torch.tensor(z['U'],device='cuda');cal=torch.tensor(calz['calibration_sum'],device='cuda');ev=torch.tensor(evz['head_writes'],device='cuda').double().sum(1);panels={'cal_centered':cal-cal.mean(0),'cal_shared_mean':cal-mu,'eval_shared_mean':ev-mu};metrics={};saved={};audits={};support={};prior=json.loads((ART/'bilin18_l17_joint_fisher_rank_curve.json').read_text())
 for name,h in panels.items():
  hc=h@B;support[name]=float((h-hc@B.T).square().sum()/h.square().sum());maps={'mean':torch.zeros_like(hc)}
  for k in cfg['ranks']:
   pc=hc.clone();pc[:,k:]=0;maps[f'PCA{k}']=pc;u=U[:,:k];maps[f'Fisher{k}']=((hc/cs)@u)@u.T*cs
  metrics[name]={};audit=0.
  for n,pred in maps.items():
   e=hc-pred;center=e-e.mean(0);mean=e.mean(0);per=.5*((e@F)*e).sum(-1);cov=.5*((center@F)*center).sum(-1).mean();bias=.5*((mean@F)*mean).sum();total=per.mean();diff=float(abs(total-cov-bias)/total) if total>1e-20 else float(abs(total-cov-bias));audit=max(audit,diff);assert diff<1e-10;metrics[name][n]={'mean_local_KL':float(total),'covariance_term':float(cov),'mean_offset_term':float(bias),'mean_fraction':float(bias/total) if total>1e-20 else None};saved[name+'_'+n]=per.cpu().numpy()
  denom=metrics[name]['mean']['mean_local_KL']
  for n in maps:metrics[name][n]['recovery']=1-metrics[name][n]['mean_local_KL']/denom
  audits[name]={'max_decomposition_relative':audit}
  print(name,{n:metrics[name][n] for n in ['mean','Fisher36','Fisher144','Fisher576']},flush=True)
 for k in cfg['ranks']:
  for method,field in [('Fisher','training_Fisher_recovery'),('PCA','training_PCA_recovery')]:assert abs(metrics['cal_centered'][f'{method}{k}']['recovery']-prior['geometry'][str(k)][field])<1e-9
 assert all(np.isfinite(v).all() for v in saved.values());np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**saved);STEM.with_suffix('.json').write_text(json.dumps({'experiment':'H518','config':cfg,'metrics':metrics,'audits':audits,'outside_supported_span_raw_fraction':support,'unique_documents':21504},indent=2)+'\n')
if __name__=='__main__':main()

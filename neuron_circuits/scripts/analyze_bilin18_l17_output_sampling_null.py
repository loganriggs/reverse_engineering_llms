"""H307 Gaussian plug-in covariance null; independent-sample sensitivity."""
from pathlib import Path
import json,hashlib
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/bilin18_tensor_decomposition';STEM=ART/'bilin18_l17_output_sampling_null'
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());path=ART/'bilin18_l17_missing_output_spectrum_statistics.npz';assert hashlib.sha256(path.read_bytes()).hexdigest()==cfg['source_sha256'];stats=np.load(path);rng=torch.Generator(device='cuda').manual_seed(cfg['seed']);out=[];checks=[]
 for group,cov in enumerate(stats['train']):
  c=torch.tensor(cov,device='cuda',dtype=torch.float64);ev=torch.linalg.eigvalsh(c).clamp_min(0);ev=ev/ev.sum();root=ev.sqrt();records={}
  for n in cfg['sample_counts']:
   sims=[]
   for rep in range(cfg['replicates']):
    x=torch.randn(n,1152,device='cuda',dtype=torch.float64,generator=rng)*root;y=torch.randn(n,1152,device='cuda',dtype=torch.float64,generator=rng)*root;a=x.T@x;b=y.T@y;e,U=torch.linalg.eigh(a);U=U.flip(1);e=e.flip(0);values={}
    for k in cfg['ranks']:
     v=U[:,:k];train=float(e[:k].sum()/a.trace());test=float(torch.einsum('ik,ij,jk->',v,b,v)/b.trace());population=float((v.square()*ev[:,None]).sum());values[str(k)]={'train':train,'test':test,'population':population,'gap':train-test}
    sims.append(values);checks.append(float(a.trace())/n)
   records[str(n)]=sims
  out.append({'group':group,'simulations':records});print('group',group,'complete',flush=True)
 observed=json.loads((ART/'bilin18_l17_output_block_transfer.json').read_text())['results'];summary={}
 for k in cfg['ranks']:
  m=np.array([[np.mean(v) for v in observed[f'block{i}']['specialist'][str(k)]] for i in range(4)]);og=float(np.diag(m).mean()-m[~np.eye(4,dtype=bool)].mean());summary[str(k)]={'observed_train':float(np.diag(m).mean()),'observed_test':float(m[~np.eye(4,dtype=bool)].mean()),'observed_gap':og,'nulls':{}}
  for n in cfg['sample_counts']:
   means={metric:[float(np.mean([g['simulations'][str(n)][rep][str(k)][metric] for g in out])) for rep in range(cfg['replicates'])] for metric in ['train','test','population','gap']};summary[str(k)]['nulls'][str(n)]={metric:{'mean':float(np.mean(v)),'replicate_range':[min(v),max(v)]} for metric,v in means.items()}
 gates={'excess_gap_both_sample_counts':all(v['observed_gap']-max(z['gap']['replicate_range'])>=.05 for v in summary.values() for z in v['nulls'].values())};STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'groups':out,'summary':summary,'mean_sample_trace_over_expected':float(np.mean(checks)),'gates':gates},indent=2)+'\n');print('FINISHED',summary,gates,flush=True)
if __name__=='__main__':main()

"""H310 matched raw versus equal-norm PCA, both evaluation metrics."""
from pathlib import Path
import json,hashlib
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/bilin18_tensor_decomposition';STEM=ART/'bilin18_l17_response_norm_control'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());path=ART/'bilin18_l17_document_repartition_responses.npy';assert hashlib.sha256(path.read_bytes()).hexdigest()==cfg['source_sha256'];y=torch.tensor(np.load(path),device='cuda').double();raw=[];unit=[];concentration=[]
 for b in range(4):
  x=y[:,b*256:(b+1)*256].reshape(18,1024,1152);energy=x.square().sum(-1);assert float(energy.min())>0;u=x/energy.sqrt()[:,:,None];raw.append(x.transpose(1,2)@x);unit.append(u.transpose(1,2)@u);sort=energy.sort(descending=True).values;concentration.append({'top1percent_share':(sort[:,:11].sum(1)/energy.sum(1)).cpu().tolist(),'top10percent_share':(sort[:,:103].sum(1)/energy.sum(1)).cpu().tolist(),'energy_effective_count':(energy.sum(1).square()/energy.square().sum(1)).cpu().tolist(),'max_share':(sort[:,0]/energy.sum(1)).cpu().tolist()})
 covs={'raw':torch.stack(raw),'unit':torch.stack(unit)};results={}
 for fitmetric in ['raw','unit']:
  results[fitmetric]={str(k):{metric:np.zeros((4,4,18)) for metric in ['raw','unit']} for k in cfg['ranks']}
  for fit in range(4):
   for g in range(18):
    _,U=torch.linalg.eigh(covs[fitmetric][fit,g]);U=U.flip(1)
    for k in cfg['ranks']:
     v=U[:,:k]
     for metric in ['raw','unit']:
      for test in range(4):c=covs[metric][test,g];results[fitmetric][str(k)][metric][fit,test,g]=float(torch.einsum('ik,ij,jk->',v,c,v)/c.trace())
   print(fitmetric,'fit',fit,'complete',flush=True)
 summary={}
 for fitmetric,rs in results.items():
  summary[fitmetric]={}
  for k,metrics in rs.items():
   summary[fitmetric][k]={}
   for metric,values in metrics.items():
    m=values.mean(2);inside=float(np.diag(m).mean());cross=float(m[~np.eye(4,dtype=bool)].mean());summary[fitmetric][k][metric]={'within':inside,'cross':cross,'gap':inside-cross};results[fitmetric][k][metric]=values.tolist()
 gates={'gap_reduction':all(summary['raw'][str(k)]['raw']['gap']-summary['unit'][str(k)]['unit']['gap']>=.1 for k in cfg['ranks']),'raw_transfer_gain':all(summary['unit'][str(k)]['raw']['cross']-summary['raw'][str(k)]['raw']['cross']>=.05 for k in cfg['ranks'])};STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'concentration':concentration,'results':results,'summary':summary,'gates':gates},indent=2)+'\n');print('FINISHED',summary,gates,flush=True)
if __name__=='__main__':main()

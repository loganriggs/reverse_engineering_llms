"""H352 frozen earlier metric versus recipient-paired and shuffled metrics."""
from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/bilin18_tensor_decomposition';STEM='bilin18_l17_head6_metric_transfer'
def main():
 cfg=json.loads((ROOT/'configs'/f'{STEM}.json').read_text())
 for k,v in cfg['sha256'].items():assert hashlib.sha256((ART/cfg[k]).read_bytes()).hexdigest()==v
 train=np.load(ART/cfg['training']);ev=np.load(ART/cfg['evaluation']);p=np.load(ART/cfg['program']);O=p['outputs'].astype('float64');G=O@O.T;inv=np.linalg.inv(G)
 def norm(M):
  tr=np.einsum('bij,ji->b',M,inv);assert tr.min()>0;return M/tr[:,None,None]
 M=norm(ev['output_metric']).reshape(128,4,16,16);T=norm(train['output_metric']).reshape(128,4,16,16).mean(0);metrics={'own':M[None],'earlier_average':np.broadcast_to(T,(1,128,4,16,16)),'shuffled':np.stack([np.roll(M,-shift,axis=0) for shift in cfg['shifts']])};rows={}
 for n in ['actual']+[f'{k}{i}' for k in ['moment','donor'] for i in range(6)]:
  c=ev[n+'_coefficients'].reshape(128,4,16);den=np.einsum('bpi,ij,bpj->bp',c,G,c)
  for key,metric in metrics.items():rows[n+'_'+key]=np.einsum('bpi,rbpij,bpj->rbp',c,metric,c)/den[None]
 rng=np.random.default_rng(cfg['bootstrap_seed']);docs=rng.integers(0,128,(4000,128));draws={n:rng.integers(0,6,(4000,6)) for n in ['moment','donor']};metricdraw=rng.integers(0,6,(4000,6));summary={};boots={}
 for key in metrics:
  A=rows['actual_'+key];summary[key]={}
  for n in ['moment','donor']:
   C=np.stack([rows[f'{n}{i}_'+key] for i in range(6)]);v=C.mean()/A.mean();b=[]
   for ix,rr,mr in zip(docs,draws[n],metricdraw):
    mm=mr if key=='shuffled' else np.array([0]);b.append(C[rr][:,mm][:,:,ix].mean()/A[mm][:,ix].mean())
   b=np.array(b);boots[key+'_'+n]=b;summary[key][n]={'ratio':float(v),'ci95':np.quantile(b,[.025,.975]).tolist()}
 contrast={n:{'own_over_shuffled':summary['own'][n]['ratio']/summary['shuffled'][n]['ratio'],'ci95':np.quantile(boots['own_'+n]/boots['shuffled_'+n],[.025,.975]).tolist()} for n in ['moment','donor']};gates={'shared_gap':all(summary['earlier_average'][n]['ci95'][0]>1.2 for n in contrast),'pairing_required':all(contrast[n]['ci95'][0]>1.2 for n in contrast)}
 np.savez_compressed(ART/f'{STEM}_rows.npz',**rows,earlier_metric=T);np.savez_compressed(ART/f'{STEM}_bootstrap.npz',**boots);(ART/f'{STEM}.json').write_text(json.dumps({'config':cfg,'summary':summary,'contrast':contrast,'gates':gates,'metric_trace_error':float(np.max(np.abs(np.einsum('bpij,ji->bp',M,inv)-1)))},indent=2)+'\n');print(summary,contrast,gates)
if __name__=='__main__':main()

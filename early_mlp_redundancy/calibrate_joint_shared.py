"""M35 training-only common loss scale; no optimizer steps or model selection."""
import json,math
from pathlib import Path
import torch
from joint_shared_loss import Experiment,P,A,sha
@torch.no_grad()
def main():
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');e=Experiment();op=P/'joint_shared_calibration.json';assert not op.exists()
 rows=[];samples=[];completed=[];progress=A/'emlp_m35_calibration_progress.pt'
 if progress.exists():
  v=torch.load(progress,weights_only=True);assert v['protocol_sha256']==sha(e.pp);rows,samples,completed,e.calls=v['rows'],v['samples'],v['completed'],v['calls']
 for entry in e.p['initial_models']:
  key=entry['kind'],entry['seed']
  if list(key) in completed:continue
  f,mu,scale=e.featurizer(entry);sampled=set()
  for assignment in [x for x in e.p['target_assignments'] if x['seed']==entry['seed']]:
   c=e.cache['inputs'][assignment['input_key']];indices=assignment['target_indices'];pred,support,_=f((c['write'].to(e.device)-mu)/scale);q=pred*scale+mu
   K,R,out,changed,Rs=e.losses(q,c,indices)
   assert torch.isfinite(K) and torch.isfinite(R)
   row=dict(kind=key[0],seed=key[1],task_update=assignment['task_update'],endpoint=assignment['endpoint'],input_key=assignment['input_key'],target_indices=indices,K=float(K),R=float(R),generator_R=[float(v) for v in Rs],active_mean=float(support.sum(-1).mean()),active_min=int(support.sum(-1).min()),active_max=int(support.sum(-1).max()))
   assert row['active_max']<=32
   rows.append(row)
   # First scheduled example in each region, no selection by loss/effect.
   region=e.targets['records'][indices[0]]['region'];assert all(e.targets['records'][i]['region']==region for i in indices)
   if region not in sampled:
    samples.append(dict(row=row,q=q.cpu(),output=out.cpu(),changed=[x.cpu() for x in changed],support=support.cpu()));sampled.add(region)
  completed.append(list(key));tmp=progress.with_suffix('.tmp');torch.save(dict(protocol_sha256=sha(e.pp),rows=rows,samples=samples,completed=completed,calls=e.calls),tmp);tmp.replace(progress)
  print('CALIBRATED',key,'ROWS',len(rows),'FORWARDS',e.calls,flush=True)
 means=[]
 for entry in e.p['initial_models']:
  rs=[x for x in rows if x['kind']==entry['kind'] and x['seed']==entry['seed']];assert len(rs)==120
  means.append(dict(kind=entry['kind'],seed=entry['seed'],mean_K=sum(x['K'] for x in rs)/120,mean_R=sum(x['R'] for x in rs)/120))
 meanK=sum(x['mean_K'] for x in means)/6;meanR=sum(x['mean_R'] for x in means)/6;assert meanR>0 and math.isfinite(meanK/meanR)
 sp=A/'emlp_m35_calibration_samples.pt';torch.save(samples,sp)
 result=dict(status='calibrated_before_training',protocol_sha256=sha(e.pp),source_sha256=sha(Path(__file__)),loss_source_sha256=sha(P.parent/'joint_shared_loss.py'),checkpoint_sha256=e.receipt.weights_sha256,rows=rows,model_means=means,pooled_K=meanK,pooled_R=meanR,response_weight=meanK/meanR,forward_count=e.calls,samples_path=str(sp),samples_sha256=sha(sp),scope='All720 scheduled training endpoints, both generators averaged per endpoint, equal weight over six M7 starts; no evaluation calibration or model fitting.')
 op.write_text(json.dumps(result,indent=2)+'\n');print('DONE WEIGHT',meanK/meanR,'MEANS',means,flush=True)
if __name__=='__main__':main()

"""Score final training-target retention; all candidates and generators retained."""
import json
from pathlib import Path
import torch
from train_write_featurizers import Featurizer
from joint_shared_loss import Experiment,P,A,sha
@torch.no_grad()
def main():
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');e=Experiment();rp=P/'joint_shared_training_results.json';train=json.loads(rp.read_text());audit=json.loads((P/'joint_shared_training_audit.json').read_text());assert audit['results_sha256']==sha(rp) and audit['status']=='passed';assert train['receipt']['protocol_sha256']==sha(e.pp);op=P/'joint_shared_retention.json';assert not op.exists();rows=[];samples=[];done=[];progress=A/'emlp_m36_retention_progress.pt'
 if progress.exists():
  saved=torch.load(progress,weights_only=True);assert saved['results_sha256']==sha(rp);rows,samples,done,e.calls=saved['rows'],saved['samples'],saved['done'],saved['calls']
 for entry in train['models']:
  key=[entry[k] for k in ['kind','seed','arm']]
  if key in done:continue
  path=Path(entry['path']);assert sha(path)==entry['sha256'];s=torch.load(path,weights_only=True);f=Featurizer(entry['kind'],entry['seed']).to(e.device);f.load_state_dict(s['state']);mu,scale=s['mean'].to(e.device),s['scale'].to(e.device);sampled=set()
  for a in [x for x in e.p['target_assignments'] if x['seed']==entry['seed']]:
   c=e.cache['inputs'][a['input_key']];pred,support,_=f((c['write'].to(e.device)-mu)/scale);K,R,out,changed,Rs=e.losses(pred*scale+mu,c,a['target_indices']);row=dict(kind=key[0],seed=key[1],arm=key[2],task_update=a['task_update'],endpoint=a['endpoint'],input_key=a['input_key'],target_indices=a['target_indices'],family=e.targets['records'][a['target_indices'][0]]['family'],K=float(K),R=float(R),generator_R=[float(x) for x in Rs],active_mean=float(support.sum(-1).mean()));rows.append(row)
   region=e.targets['records'][a['target_indices'][0]]['region']
   if region not in sampled:samples.append(dict(row=row,output=out.cpu(),changed=[x.cpu() for x in changed]));sampled.add(region)
  done.append(key);tmp=progress.with_suffix('.tmp');torch.save(dict(results_sha256=sha(rp),rows=rows,samples=samples,done=done,calls=e.calls),tmp);tmp.replace(progress);print('RETENTION',key,flush=True)
 summaries=[]
 for entry in train['models']:
  rs=[x for x in rows if all(x[k]==entry[k] for k in ['kind','seed','arm'])]
  for family in [None]+sorted({x['family'] for x in rs}):
   sub=rs if family is None else [x for x in rs if x['family']==family]
   summaries.append({k:entry[k] for k in ['kind','seed','arm']}|dict(family=family,count=len(sub),mean_K=sum(x['K'] for x in sub)/len(sub),mean_R=sum(x['R'] for x in sub)/len(sub),active_mean=sum(x['active_mean'] for x in sub)/len(sub)))
 sp=A/'emlp_m36_retention_samples.pt';torch.save(samples,sp);result=dict(results_sha256=sha(rp),source_sha256=sha(Path(__file__)),loss_source_sha256=sha(P.parent/'joint_shared_loss.py'),rows=rows,summaries=summaries,forward_count=e.calls,samples_path=str(sp),samples_sha256=sha(sp),scope='Final familiar training-target retention only; no held-out native transfer/output evaluation yet.')
 op.write_text(json.dumps(result,indent=2)+'\n')
 for x in summaries:
  if x['family'] is None:print(x,flush=True)
if __name__=='__main__':main()

"""Independent M9 NumPy metrics and every family/pair aggregate."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parent/'validation';torch.set_num_threads(4)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def lse(x):
 m=np.max(x);return float(m+np.log(np.exp(x-m).sum()))
def logsoft(x):
 x=x.astype(float);x-=x.max(-1,keepdims=True);return x-np.log(np.exp(x).sum(-1,keepdims=True))
rp=P/'featurizer_task_transfer.json';r=json.loads(rp.read_text());assert sha(P/'featurizer_task_transfer_protocol.json')==r['protocol_sha256'];sp=Path(r['samples_path']);assert sha(sp)==r['samples_sha256'];samples=torch.load(sp,weights_only=True);errors=[];families=set()
for s in samples:
 row=r['rows'][s['row']];families.add(row['family']);native=s['native'].numpy();out=s['output'].numpy();lp=logsoft(native);lq=logsoft(out);kl=(np.exp(lp)*(lp-lq)).sum(-1);ids=s['ids'].numpy();ce=np.take_along_axis(lp[:,:-1]-lq[:,:-1],ids[:,1:,None],axis=-1).mean();vals=dict(mean_KL=kl.mean(),last_KL=kl[0,-1],max_position_KL=kl.max(),observed_next_token_dCE=ce,top1_agreement=(native.argmax(-1)==out.argmax(-1)).mean());v=s['native_product'].numpy().astype(float);q=s['reconstructed_product'].numpy().astype(float);vals['local_relative_error']=np.sum((q-v)**2)/np.sum(v*v)
 for k,value in vals.items():errors.append(abs(float(value)-row[k]));assert errors[-1]<1e-7,(k,errors[-1])
 margins=[]
 for field,lg in [('native_task',native),('task',out)]:
  last=lg[0,-1].astype(float);groups=s['groups'];variant=s['variant'];margin=lse(last[groups[variant]])-lse(last[groups[1-variant]]);margins.append(margin);task=row[field];assert abs(margin-task['margin'])<1e-10;assert bool(margin>0)==task['binary'];assert int(last.argmax())==task['top_token'];assert (int(last.argmax()) in groups[variant])==task['top1']
 assert abs(abs(margins[1]-margins[0])-row['absolute_margin_error'])<1e-10
for summary in r['summaries']:
 rs=[x for x in r['rows'] if all(x[k]==summary[k] for k in ['kind','seed','objective']) and (summary['family'] is None or x['family']==summary['family'])];assert len(rs)==summary['endpoints'];pairs={x['id'] for x in rs};assert len(pairs)==summary['pairs']
 for k in ['mean_KL','last_KL','observed_next_token_dCE','top1_agreement','local_relative_error','absolute_margin_error','binary_decision_agreement']:assert abs(np.mean([x[k] for x in rs])-summary[k])<1e-10
 for field,prefix in [('task',''),('native_task','native_')]:
  for key,name in [('binary','binary_rate'),('top1','task_top1_rate')]:assert abs(np.mean([x[field][key] for x in rs])-summary[prefix+name])<1e-10
  both=np.mean([all(x[field]['binary'] for x in rs if x['id']==pair) for pair in pairs]);assert abs(both-summary[prefix+'both_endpoint_binary_rate'])<1e-10
for macro in r['equal_family_macros']:
 rs=[s for s in r['summaries'] if s['family'] is not None and all(s[k]==macro[k] for k in ['kind','seed','objective'])];assert len(rs)==10
 for k in ['mean_KL','last_KL','binary_rate','task_top1_rate','both_endpoint_binary_rate','absolute_margin_error']:assert abs(np.mean([s[k] for s in rs])-macro[k])<1e-10
assert len(families)==10
out=dict(status='passed',full_output_sample_checks=len(samples),families=len(families),max_absolute_metric_error=max(errors),all1280records_and88summaries_aggregated=True,equal_family_macros_checked=True,result_sha256=sha(rp),scope='NumPy metrics/answer-group scoring foroneendpoint per family×8models; allfamily/pair aggregate receipts independently checked. Frozen model forwards were audited inM8, not independently reimplemented here.')
(P/'featurizer_task_transfer_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(out)

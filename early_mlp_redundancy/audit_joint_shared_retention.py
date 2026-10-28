"""Full retention accounting and independent sampled NumPy metric replay."""
from pathlib import Path
import hashlib,json,math
import numpy as np
import torch
P=Path(__file__).resolve().parent/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b):assert math.isclose(a,b,rel_tol=1e-8,abs_tol=1e-10),(a,b)
def lp(t):
 x=t.numpy().astype(float);x-=x.max(-1,keepdims=True);return x-np.log(np.exp(x).sum(-1,keepdims=True))
rp=P/'joint_shared_retention.json';r=json.loads(rp.read_text());trp=P/'joint_shared_training_results.json';train=json.loads(trp.read_text());assert r['results_sha256']==sha(trp);assert r['source_sha256']==sha(P.parent/'evaluate_joint_shared_retention.py');assert r['loss_source_sha256']==sha(P.parent/'joint_shared_loss.py');p=json.loads((P/'joint_shared_comparison_protocol.json').read_text());tp=P/'response_training_targets.json';assert sha(tp)==p['target_source_sha256'];t=json.loads(tp.read_text());assert sha(Path(t['cache_path']))==t['cache_sha256'];cache=torch.load(t['cache_path'],weights_only=True);sp=Path(r['samples_path']);assert sha(sp)==r['samples_sha256'];samples=torch.load(sp,weights_only=True)
assign={(x['seed'],x['task_update'],x['endpoint']):x for x in p['target_assignments']};seen=set();calls=0
for row in r['rows']:
 key=tuple(row[k] for k in ['kind','seed','arm','task_update','endpoint']);assert key not in seen;seen.add(key);a=assign[row['seed'],row['task_update'],row['endpoint']];assert row['input_key']==a['input_key'] and row['target_indices']==a['target_indices'];assert row['family']==t['records'][a['target_indices'][0]]['family'];close(row['R'],sum(row['generator_R'])/2);assert all(math.isfinite(row[k]) and row[k]>=-1e-10 for k in ['K','R']);calls+=1+sum(not t['records'][i]['noop'] for i in a['target_indices'])
assert len(seen)==1440 and r['forward_count']==calls==3978
for m in train['models']:assert len([x for x in r['rows'] if all(x[k]==m[k] for k in ['kind','seed','arm'])])==120
assert len(r['summaries'])==132
for s in r['summaries']:
 rows=[x for x in r['rows'] if all(x[k]==s[k] for k in ['kind','seed','arm']) and (s['family'] is None or x['family']==s['family'])];assert len(rows)==s['count']
 for field,source in [('mean_K','K'),('mean_R','R'),('active_mean','active_mean')]:close(sum(x[source] for x in rows)/len(rows),s[field])
assert len(samples)==24;errors=[]
for sample in samples:
 row=sample['row'];assert row in r['rows'];native=lp(cache['inputs'][row['input_key']]['native']);out=lp(sample['output']);K=float((np.exp(native)*(native-out)).sum(-1).mean());close(K,row['K']);errors.append(abs(K-row['K']))
 for i,v,saved in zip(row['target_indices'],sample['changed'],row['generator_R']):
  nr=lp(cache['targets'][i]['native_changed'])-native[0,-1];fr=lp(v)-out[0,-1];R=float((np.exp(native[0,-1])*(fr-nr)**2).sum());close(R,saved);errors.append(abs(R-saved))
  if t['records'][i]['noop']:assert torch.equal(v,sample['output'][0,-1])
out=dict(status='passed',result_sha256=sha(rp),scheduled_rows=1440,aggregate_checks=132,sampled_output_checks=24,sampled_response_checks=48,max_sample_metric_error=max(errors),scope='All schedule/aggregate arithmetic plus independent sampled full-vector metrics; not full native reexecution or generalization validation.')
(P/'joint_shared_retention_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(out)

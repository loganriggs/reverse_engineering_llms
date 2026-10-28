"""Independent partition, coverage and full-position logit audit for deletion diagnostic."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parent
path=P/'validation/mlp1_group_deletion.json';r=json.loads(path.read_text());sets=r['removed_sets']
quarters=[set(sets[f'quarter_{i}']) for i in range(4)]
assert all(len(q)==1152 for q in quarters) and len(set.union(*quarters))==4608
assert set(sets['half_01'])==quarters[0]|quarters[1] and set(sets['half_23'])==quarters[2]|quarters[3]
assert set(sets['all_products'])==set(range(4608))
assert len(r['rows'])==1280 and len(r['native'])==160 and r['forward_count']==1441 and r['identity_max_logit_error']==0
p7=json.loads((P/'validation/training_task_responses.json').read_text());old={(x['id'],i):e for x in p7['native'] for i,e in enumerate(x['endpoints'])}
for x in r['native']:
 prev=old[(x['id'],x['endpoint'])]
 for f in ['binary','top1','top_token','margin']:assert x[f]==prev[f]
 rr=[z for z in r['rows'] if (z['id'],z['endpoint'])==(x['id'],x['endpoint'])]
 assert len(rr)==8 and {z['kind'] for z in rr}==set(sets)
errors=[]
def logsoft(x):
 z=x-x.max(-1,keepdims=True);return z-np.log(np.exp(z).sum(-1,keepdims=True))
for sample in torch.load('/tmp/emlp_d1_deletion_logits.pt',weights_only=True):
 row=r['rows'][sample['row']];a=sample['native'].double().numpy();b=sample['edited'].double().numpy();ids=sample['ids'].numpy();p=logsoft(a);q=logsoft(b);k=(np.exp(p)*(p-q)).sum(-1)
 target=ids[:,1:,None];ce=(-np.take_along_axis(q[:,:-1],target,-1)+np.take_along_axis(p[:,:-1],target,-1)).mean()
 expected=dict(mean_KL=k.mean(),max_position_KL=k.max(),last_KL=k[0,-1],observed_next_token_dCE=ce,top1_agreement=(a.argmax(-1)==b.argmax(-1)).mean())
 for f,v in expected.items():errors.append(abs(float(v)-row[f]))
for summary in r['summaries']:
 rr=[x for x in r['rows'] if x['kind']==summary['kind'] and (summary['family'] is None or x['family']==summary['family'])]
 assert len(rr)==summary['n']
 for f in ['mean_KL','last_KL','observed_next_token_dCE','top1_agreement']:
  assert abs(sum(x[f] for x in rr)/len(rr)-summary[f])<1e-12
assert max(errors)<1e-7
out=dict(status='passed',partition_exact=True,native_endpoint_replays=160,sampled_full_sequence_outputs=8,max_metric_error=max(errors),result_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
(P/'validation/mlp1_group_deletion_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

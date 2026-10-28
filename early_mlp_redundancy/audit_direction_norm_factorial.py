"""Audit P9 against P8 and independently replay sampled metrics."""
import json,hashlib
from pathlib import Path
import torch
import numpy as np
from audit_preamble_screen import kl,margin
P=Path(__file__).resolve().parent
p8=P/'validation/preamble_transfer.json';p9=P/'validation/direction_norm_factorial.json'
a=json.loads(p8.read_text());b=json.loads(p9.read_text())
def key(r):return r['id'],r['preamble_index'],r['receiver']
old={key(r):r for r in a['rows'] if r['kind']=='context_specific'}
new={key(r):r for r in b['rows'] if r['kind']=='context_specific'}
assert old.keys()==new.keys() and len(new)==96
fields=['remaining_donor_KL','native_donor_KL','donor_margin_gap','margin_gain','old_norm','new_norm','response_cosine']
replay=max(abs(old[k][f]-new[k][f]) for k in old for f in fields);assert replay==0
assert b['forward_count']==576 and len(b['rows'])==192
assert {key(r) for r in b['rows'] if r['kind']=='new_direction_old_norm'}==old.keys()
samples=torch.load('/tmp/emlp_p9_factorial_samples.pt',weights_only=True);errors=[]
for x in samples['samples']:
 r=b['rows'][x['row']];teacher=x['teacher'].double().numpy();base=x['baseline'].double().numpy();cand=x['candidate'].double().numpy()
 errors.extend([abs(kl(teacher,cand)-r['remaining_donor_KL']),abs(kl(teacher,base)-r['native_donor_KL']),abs(margin(cand,x['groups'],x['variant'])-margin(base,x['groups'],x['variant'])-r['margin_gain'])])
for c in b['cohorts']:
 for kind,v in c['responses'].items():
  rr=[r for r in b['rows'] if (r['family'],r['preamble_index'],r['kind'])==(c['family'],c['preamble_index'],kind)]
  assert len(rr)==v['n']
  assert abs(sum(r['remaining_donor_KL'] for r in rr)/sum(r['native_donor_KL'] for r in rr)-v['remaining_donor_KL_ratio'])<1e-12
  assert abs(sum(r['margin_gain'] for r in rr)/sum(r['donor_margin_gap'] for r in rr)-v['margin_recovery'])<1e-12
assert max(errors)<1e-10
out=dict(status='passed',native_response_replays=96,replay_max_error=replay,sampled_logit_rows=len(samples['samples']),sample_metric_max_error=max(errors),p8_sha256=hashlib.sha256(p8.read_bytes()).hexdigest(),p9_sha256=hashlib.sha256(p9.read_bytes()).hexdigest())
(P/'validation/direction_norm_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

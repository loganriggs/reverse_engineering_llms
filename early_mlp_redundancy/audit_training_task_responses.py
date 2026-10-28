"""Audit P7 training membership, cohort arithmetic, and saved logit metrics."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parent
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--samples',type=Path,default=Path('/tmp/emlp_p7_logits.pt'));args=ap.parse_args()
f=P/'validation/training_task_responses.json';d=json.loads(f.read_text());panel=json.loads((P/'validation/training_source_context_audit.json').read_text())
assert d['panel_sha256']==hashlib.sha256((P/'validation/training_source_context_audit.json').read_bytes()).hexdigest()
assert d['source_sha256']==hashlib.sha256((P/'probe_training_task_responses.py').read_bytes()).hexdigest()
assert len(panel['rows'])==len(d['native'])==80 and len(d['rows'])==760 and d['forward_count']==1072
assert {r['id'] for r in panel['rows']}=={r['id'] for r in d['native']}
assert {r['id'] for r in d['rows']}=={r['id'] for r in panel['rows'] if r['single_token_edit']}
assert not set(d['families'])&{'sequence_copy','spatial_inverse','weekday_successor','role_binding','tense'}
maxratio=0.
for fam,s in d['families'].items():
 nn=[n for n in d['native'] if n['family']==fam]
 binary=np.mean([n['both_binary'] for n in nn]);top=np.mean([n['top1_rate'] for n in nn])
 assert binary==s['both_endpoint_binary_rate'] and top==s['full_vocab_top1_rate']
 assert s['competence_gate_pass']==bool(binary>=.75 and top>=.5)
 for kind,summary in s['responses'].items():
  rr=[r for r in d['rows'] if r['family']==fam and r['kind']==kind]
  value=sum(r['remaining_donor_KL'] for r in rr)/sum(r['native_donor_KL'] for r in rr)
  maxratio=max(maxratio,abs(value-summary['remaining_donor_KL_ratio']))
  assert len(rr)==summary['n']
def lse(x):
 v=x.max();return float(v+np.log(np.exp(x-v).sum()))
def logp(t):
 a=t.double().numpy();return a-lse(a)
maxkl=0.;maxmargin=0.
for s in torch.load(args.samples,map_location='cpu',weights_only=True):
 row=d['rows'][s['row']];a=logp(s['teacher']);b=logp(s['baseline']);c=logp(s['candidate'])
 for val,k in [((np.exp(a)*(a-b)).sum(),'native_donor_KL'),((np.exp(a)*(a-c)).sum(),'remaining_donor_KL'),((np.exp(b)*(b-c)).sum(),'recipient_change_KL')]:maxkl=max(maxkl,abs(float(val)-row[k]))
 groups=s['groups'];v=s['variant']
 def margin(x):return lse(x[groups[v]])-lse(x[groups[1-v]])
 for val,k in [(margin(a)-margin(b),'donor_margin_gap'),(margin(c)-margin(b),'margin_gain')]:maxmargin=max(maxmargin,abs(val-row[k]))
assert maxkl<1e-10 and maxmargin<1e-10
out=dict(status='passed',result_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),samples_sha256=hashlib.sha256(args.samples.read_bytes()).hexdigest(),training_pairs_checked=80,
 single_token_pairs_checked=76,all_cohort_competence_gates_recomputed=True,max_ratio_error=maxratio,max_numpy_KL_error=maxkl,max_numpy_margin_error=maxmargin,
 scope='All membership/coverage/cohort aggregates; independent KL and margin calculations for2saved directions. No model rerun or reserved task evaluation.',new_model_forwards=0)
(P/'validation/training_task_responses_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

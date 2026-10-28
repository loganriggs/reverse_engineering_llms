"""Audit P6 saved logits, prior full-response replay, and summary arithmetic."""
import argparse,json,hashlib
from pathlib import Path
import torch
import numpy as np
P=Path(__file__).resolve().parent
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--samples',type=Path,default=Path('/tmp/emlp_p6_logits.pt'));args=ap.parse_args()
f=P/'validation/response_positions.json';d=json.loads(f.read_text());old=json.loads((P/'validation/prefix_response_factorial.json').read_text())
assert len(d['rows'])==224 and d['forward_count']==320 and d['identity_error']==d['response_before_changed_word_error']==0
assert d['source_sha256']==hashlib.sha256((P/'probe_response_positions.py').read_bytes()).hexdigest()
maxmean=0.;replay=0.;maxkl=0.;count=0
for arm,s in d['summary'].items():
 rr=[r for r in d['rows'] if r['arm']==arm]
 for region in ['at_edit','future']:
  mean=float(np.mean([r['donor_gap_'+region] for r in rr]));maxmean=max(maxmean,abs(mean-s['mean_donor_gap'][region]))
  ratio=sum(r['donor_gap_'+region] for r in rr)/sum(r['baseline_gap_'+region] for r in rr)
  assert abs(ratio-s['remaining_donor_gap'][region])<1e-14
prior={(r['context'],r['anchor'],r['receiver']):r for r in old['rows'] if r['kind']=='context'}
for r in d['rows']:
 if r['arm']=='all_positions':
  q=prior[r['context'],r['anchor'],r['receiver']]['metrics']['response_only']
  for region in ['at_edit','future']:replay=max(replay,abs(q['donor_gap_'+region]-r['donor_gap_'+region]))
def logp(t):
 x=t.double().numpy();x=x-x.max(-1,keepdims=True);return x-np.log(np.exp(x).sum(-1,keepdims=True))
for s in torch.load(args.samples,map_location='cpu',weights_only=True):
 row=d['rows'][s['row_index']];b=logp(s['edited'])
 for ref,prefix in [('donor','donor_gap'),('base','recipient_change')]:
  a=logp(s[ref]);v=(np.exp(a)*(a-b)).sum(-1)[0]
  for region,value in [('at_edit',v[0]),('future',v[1:].mean())]:
   maxkl=max(maxkl,abs(float(value)-row[prefix+'_'+region]));count+=1
# Expected source-context repetitions are a design property: all variation across anchors follows changed word.
for ci in range(4):
 for rec in (0,1):
  rr=[r for r in d['directions'] if r['context']==ci and r['receiver']==rec]
  prefixes=[r['recipient_ids'][0][:r['changed_position']+1] for r in rr]
  assert all(x==prefixes[0] for x in prefixes)
assert maxmean<1e-14 and replay==0 and maxkl<1e-10
out=dict(status='passed',result_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),samples_sha256=hashlib.sha256(args.samples.read_bytes()).hexdigest(),
 prior_full_response_replay_max_error=replay,max_mean_error=maxmean,max_numpy_KL_error=maxkl,independent_scalar_metrics_checked=count,
 source_context_design='Four anchors share identical causal prefix at changed word; at most8source receiver/donor contexts, not32independent directions.',
 scope='Overall means/ratios; all32full-response prior arms;7saved-logit examples from first direction; source-prefix identity checked from tokenIDs.',new_model_forwards=0)
(P/'validation/response_positions_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

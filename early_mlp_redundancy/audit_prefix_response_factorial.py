"""Independent saved-logit and aggregate audit of EMLP-P5."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parent
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--samples',type=Path,default=Path('/tmp/emlp_p5_logits.pt'));args=ap.parse_args()
f=P/'validation/prefix_response_factorial.json';d=json.loads(f.read_text())
assert d['forward_count']==704 and len(d['rows'])==160 and d['identity_error']==d['joint_replay_error']==0
assert d['source_sha256']==hashlib.sha256((P/'probe_prefix_response_factorial.py').read_bytes()).hexdigest()
assert d['panel_source_sha256']==hashlib.sha256((P/'probe_mlp2_token_transfer.py').read_bytes()).hexdigest()
maxmean=0.;maxkl=0.;count=0
for kind,s in d['summary'].items():
 rr=[r for r in d['rows'] if r['kind']==kind]
 for a,mm in s['means'].items():
  for k,v in mm.items():maxmean=max(maxmean,abs(float(np.mean([r['metrics'][a][k] for r in rr]))-v))
  for region in ['at_edit','future']:
   ratio=sum(r['metrics'][a]['donor_gap_'+region] for r in rr)/sum(r['metrics']['baseline']['donor_gap_'+region] for r in rr)
   assert abs(ratio-s['remaining_donor_gap'][a][region])<1e-14

def logp(t):
 a=t.double().numpy();a-=a.max(-1,keepdims=True);return a-np.log(np.exp(a).sum(-1,keepdims=True))
for sample in torch.load(args.samples,map_location='cpu',weights_only=True):
 r=d['rows'][sample['row_index']]
 for reference,prefix in [(sample['donor'],'donor_gap'),(sample['outputs']['baseline'],'recipient_change')]:
  a=logp(reference)
  for arm,t in sample['outputs'].items():
   b=logp(t);kl=(np.exp(a)*(a-b)).sum(-1)[0]
   for suffix,value in [('at_edit',kl[0]),('future',kl[1:].mean())]:
    maxkl=max(maxkl,abs(float(value)-r['metrics'][arm][prefix+'_'+suffix]));count+=1
prior=json.loads((P/'validation/mlp1_mediation.json').read_text());old={(r['context'],r['anchor'],r['receiver']):r for r in prior['rows'] if r['arm']=='prefix_MLP0'}
replay=0.
for r in d['rows']:
 if r['kind']=='context':
  o=old[r['context'],r['anchor'],r['receiver']]
  for region in ['at_edit','future']:replay=max(replay,abs(r['metrics']['joint']['donor_gap_'+region]-o['remaining_donor_gap_KL_'+region]))
assert maxmean<1e-14 and maxkl<1e-10 and replay<1e-6
out=dict(status='passed',result_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),samples_sha256=hashlib.sha256(args.samples.read_bytes()).hexdigest(),
 max_mean_error=maxmean,max_independent_numpy_KL_error=maxkl,independent_scalar_metrics_checked=count,
 prior_prefix_arm_KL_replay_max_absolute_error=replay,new_model_forwards=0,
 scope='All overall means/ratio arithmetic; independent KL from 2saved direction examples; prior P4 prefix-arm comparison. Addition-versus-assignment float32 differences retained.')
(P/'validation/prefix_response_factorial_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

"""Independent NumPy audit of EMLP-P2 saved logits and every summary row."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parent
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--samples',type=Path,default=Path('/tmp/emlp_p2_audit_logits.pt'))
args=ap.parse_args();path=P/'validation/mlp2_token_transfer.json';d=json.loads(path.read_text())
assert d['forward_count']==704 and len(d['rows'])==160
assert d['source_sha256']==hashlib.sha256((P/'probe_mlp2_token_transfer.py').read_bytes()).hexdigest()
maxmean=0.;maxcausal=0.
for kind in ['context','signed_permutation']:
 rows=[r for r in d['rows'] if r['kind']==kind]
 for arm in d['arms']:
  for region in ['at_edit_KL','future_mean_KL','last_KL']:
   mean=float(np.mean([r['metrics'][arm][region] for r in rows]));maxmean=max(maxmean,abs(mean-d['summary'][kind]['means'][arm][region]))
 for r in rows:
  for arm in d['arms']:
   v=np.array(r['metrics'][arm]['per_position_KL']);assert np.isfinite(v).all() and (v>=-1e-12).all()
   assert abs(v[1:].mean()-r['metrics'][arm]['future_mean_KL'])<1e-15
  maxcausal=max(maxcausal,abs(r['metrics']['live']['at_edit_KL']-r['metrics']['local_only']['at_edit_KL']),abs(r['metrics']['frozen_all']['at_edit_KL']-r['metrics']['future_only']['at_edit_KL']))
def logp(t):
 x=t.double().numpy();x=x-x.max(-1,keepdims=True);return x-np.log(np.exp(x).sum(-1,keepdims=True))
maxkl=0.;checked=0
for sample in torch.load(args.samples,map_location='cpu',weights_only=True):
 row=d['rows'][sample['row_index']];a=logp(sample['base'])
 for arm,t in sample['outputs'].items():
  b=logp(t);v=(np.exp(a)*(a-b)).sum(-1)[0]
  maxkl=max(maxkl,float(np.abs(v-np.array(row['metrics'][arm]['per_position_KL'])).max()));checked+=len(v)
assert maxkl<1e-10 and maxmean<1e-15 and maxcausal==0
out=dict(status='passed',result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),sample_sha256=hashlib.sha256(args.samples.read_bytes()).hexdigest(),
 max_summary_mean_error=maxmean,max_independent_numpy_KL_error=maxkl,independently_checked_position_KLs=checked,max_causal_factorial_equalities_error=maxcausal,
 new_model_forwards=0,scope='All overall summary rows, per-position means/finiteness, and edited-position causal equalities; independent logits for first context direction and first random control only.')
(P/'validation/mlp2_token_transfer_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

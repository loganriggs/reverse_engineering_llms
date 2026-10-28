"""Recompute every saved EMLP-P4 vector metric independently with NumPy."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parent
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--vectors',type=Path,default=Path('/tmp/emlp_p4_vectors.pt'));args=ap.parse_args()
f=P/'validation/mlp1_mediation.json';d=json.loads(f.read_text());vectors=torch.load(args.vectors,map_location='cpu',weights_only=True)
assert len(vectors)==len(d['rows'])==224 and d['forward_count']==288
assert d['identity_max_error']==d['oracle_relative_error']==0
assert d['source_sha256']==hashlib.sha256((P/'probe_mlp1_mediation.py').read_bytes()).hexdigest()
assert d['panel_source_sha256']==hashlib.sha256((P/'probe_mlp2_token_transfer.py').read_bytes()).hexdigest()
maxrel=0.;aggregate={}
for item in vectors:
 r=d['rows'][item['row']];t=item['target'].double().numpy().ravel();p=item['response'].double().numpy().ravel()
 tt=float(t@t);pp=float(p@p);dot=float(t@p);ee=float(np.square(t-p).sum())
 metrics=dict(target_sqnorm=tt,response_sqnorm=pp,dot=dot,error_sqnorm=ee,signed_projection=dot/tt,cosine=dot/np.sqrt(tt*pp),norm_ratio=np.sqrt(pp/tt),relative_L2=np.sqrt(ee/tt))
 for k,v in metrics.items():maxrel=max(maxrel,abs(v-r['vector'][k])/max(abs(v),1e-12))
 aggregate.setdefault(r['arm'],[]).append((tt,pp,dot,ee))
maxsummary=0.
for arm,arr in aggregate.items():
 tt,pp,dot,ee=np.array(arr).sum(0)
 scores=dict(signed_projection=dot/tt,cosine=dot/np.sqrt(tt*pp),norm_ratio=np.sqrt(pp/tt),relative_L2=np.sqrt(ee/tt),write_R2=1-ee/tt)
 for k,v in scores.items():maxsummary=max(maxsummary,abs(v-d['summary'][arm][k]))
 for region in ['at_edit','future']:
  rr=[r for r in d['rows'] if r['arm']==arm]
  value=sum(r['remaining_donor_gap_KL_'+region] for r in rr)/sum(r['native_donor_gap_KL_'+region] for r in rr)
  assert abs(value-d['summary'][arm]['relative_remaining_donor_KL'][region])<1e-14
assert maxrel<1e-10 and maxsummary<1e-12
out=dict(status='passed',result_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),vectors_sha256=hashlib.sha256(args.vectors.read_bytes()).hexdigest(),
 vector_rows_checked=len(vectors),max_relative_metric_error=maxrel,max_absolute_summary_error=maxsummary,
 scope='All vector metrics independently replayed; overall vector aggregates and stored KL-ratio arithmetic checked. Historical/native logits not recomputed in this audit.',new_model_forwards=0)
(P/'validation/mlp1_mediation_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

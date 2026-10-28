"""M15 NumPy native answer/gate audit for all56newcontexts."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parent/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def lse(x):
 m=x.max();return float(m+np.log(np.exp(x-m).sum()))
ip=P/'context_competence.json';r=json.loads(ip.read_text());pp=P/'context_challenge_protocol.json';assert sha(pp)==r['protocol_sha256'];sp=Path(r['samples_path']);assert sha(sp)==r['samples_sha256'];samples=torch.load(sp,weights_only=True);errors=[]
for s in samples:
 x=s['logits'].numpy().astype(float);v=s['variant'];g=s['groups'];margin=lse(x[g[v]])-lse(x[g[1-v]]);row=next(x for x in r['results'] if x['context']==s['context'] and x['id']==s['id'])['endpoints'][s['endpoint']];errors.append(abs(margin-row['margin']));assert errors[-1]<1e-10;assert bool(margin>0)==row['binary'];assert (int(x.argmax()) in g[v])==row['top1']
for c in r['cohorts']:
 rows=[x for x in r['results'] if x['context']==c['context'] and x['family']==c['family']];binary=np.mean([all(e['binary'] for e in x['endpoints']) for x in rows]);top=np.mean([e['top1'] for x in rows for e in x['endpoints']]);assert binary==c['both_binary_rate'] and top==c['top1_rate'];assert bool(binary>=.75 and top>=.5)==c['pass_gate']
out=dict(status='passed',native_endpoint_checks=len(samples),cohort_checks=len(r['cohorts']),max_margin_error=max(errors),result_sha256=sha(ip));(P/'context_competence_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(out)

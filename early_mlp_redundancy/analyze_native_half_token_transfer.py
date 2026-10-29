"""M45 coverage-stratified comparisons with no outcome-based model selection."""
from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent/'validation'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rp=P/'native_half_token_transfer.json';r=json.loads(rp.read_text());assert json.loads((P/'native_half_token_transfer_audit.json').read_text())['result_sha256']==sha(rp);summaries=[]
for domain,part in [('task','train'),('task','evaluation'),('natural','evaluation')]:
 for layer in [0,1,2]:
  for arm in ['rank32','token_mean','token_centered_rank32']:
   for half in [None,'half_01','half_23']:
    rr=[x for x in r['rows'] if (x['domain'],x['split'],x['layer'],x['arm'])==(domain,part,layer,arm) and (half is None or x['removed']==half)];n=len(rr);s=dict(domain=domain,split=part,layer=layer,arm=arm,removed=half,n=n,mean_KL=sum(x['mean_KL'] for x in rr)/n,covered_positions=sum(x['covered_positions'] for x in rr),positions=sum(x['positions'] for x in rr))
    for label in ['covered','uncovered']:
     err=sum(x[label+'_error_energy'] for x in rr);en=sum(x[label+'_net_energy'] for x in rr);s[label+'_relative_error']=err/en if en else None
    s['coverage']=s['covered_positions']/s['positions'];summaries.append(s)
    if half is None:print(s)
(P/'native_half_token_comparison.json').write_text(json.dumps(dict(result_sha256=sha(rp),source_sha256=sha(Path(__file__)),summaries=summaries,scope='Covered/uncovered local errors pooled by native full product-write energy; output KL across all positions, not restricted to covered positions. Counts include both dependent masks. All rows/families retained in source data.'),indent=2)+'\n')

"""M23 fixed paired comparisons and equal-family causal sensitivities."""
from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
tp=P/'nomomentum_training_results.json';cp=P/'nomomentum_native_fidelity.json';t=json.loads(tp.read_text());c=json.loads(cp.read_text());rows=[]
for kind in ['scalar','block']:
 for seed in [71,72]:
  for region in ['prefix','task']:
   item=dict(kind=kind,seed=seed,region=region,arms={})
   for arm in ['initial','intact_KL','intact_KL_plus_response']:
    base=next(x for x in c['summaries'] if (x['kind'],x['seed'],x['region'],x['arm'],x['quadrant'],x['family'])==(kind,seed,region,arm,'all',None));families=[x for x in c['summaries'] if (x['kind'],x['seed'],x['region'],x['arm'],x['quadrant'])==(kind,seed,region,arm,'all') and x['family'] is not None];valid=[x for x in families if x['relative_margin_error'] is not None];validlog=[x for x in families if x['relative_logprob_response_error'] is not None]
    item['arms'][arm]=dict(relative_margin_error=base['relative_margin_error'],relative_logprob_response_error=base['relative_logprob_response_error'],equal_family_margin_error=sum(x['relative_margin_error'] for x in valid)/len(valid),equal_family_logprob_error=sum(x['relative_logprob_response_error'] for x in validlog)/len(validlog),margin_families=len(valid),logprob_families=len(validlog),zero_margin_families=[x['family'] for x in families if x['relative_margin_error'] is None])
   rows.append(item)
out=dict(scope='Predeclared weighted and equal-family summaries, no refits or selections. Ratios may be large for small family-native effects; null denominators excluded only from equal-family mean and explicitly counted.',training_results_sha256=sha(tp),causal_results_sha256=sha(cp),rows=rows)
(P/'nomomentum_fidelity_comparison.json').write_text(json.dumps(out,indent=2)+'\n')
for r in rows:print(r['kind'],r['seed'],r['region'],'equal-family margin',*[round(r['arms'][a]['equal_family_margin_error'],3) for a in ['initial','intact_KL','intact_KL_plus_response']])
print('BOTH-HELDOUT MARGIN')
for x in c['summaries']:
 if x['quadrant']=='evaluation:evaluation' and x['family'] is None:print(x['kind'],x['seed'],x['arm'],x['region'],round(x['relative_margin_error'],4))

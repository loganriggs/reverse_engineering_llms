"""M22 bounded step comparisons, no optimizer selection or retraining."""
from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent/'validation';ip=P/'response_adam_replay.json';r=json.loads(ip.read_text());rows=[]
for step in [79,80,81,82,83]:
 actual=r['records'][step];variants=[x for x in r['counterfactuals'] if x['step']==step];rows.append(dict(step=step,domain=actual['update_domain'],family=actual['family'],before=actual['sentinel_before'],actual_after=actual['sentinel_after'],actual_change=actual['total_change'],actual_first_order=actual['first_order_prediction'],actual_normalized_update_norm=actual['parameter_delta_norm'],variants={v['variant']:v for v in variants}))
out=dict(scope='Five post-selected diagnostic steps of one failed trajectory. Same actual secondmoment/bias corrections; current-only norm-matched before unit-column normalization, achieved postnormalization norms reported. One-step effects, not a demonstrated stable training algorithm.',replay_sha256=hashlib.sha256(ip.read_bytes()).hexdigest(),rows=rows);(P/'response_adam_analysis.json').write_text(json.dumps(out,indent=2)+'\n')
for x in rows:print(x['step'],x['domain'],x['family'],'before/actual',round(x['before'],6),round(x['actual_after'],6),[(name,round(v['sentinel_loss'],6),round(v['first_order_prediction'],6),round(v['normalized_update_norm'],6)) for name,v in x['variants'].items()])

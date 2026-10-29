"""M40 descriptive matched grouping comparisons; no candidate selection."""
from pathlib import Path
import json, hashlib
P=Path(__file__).resolve().parent
V=P/'validation'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rp=V/'coactivation_outputs.json';r=json.loads(rp.read_text())
a=json.loads((V/'coactivation_outputs_audit.json').read_text());assert a['status']=='passed' and a['result_sha256']==sha(rp)
rows=[]
for seed in [71,72]:
 for arm in ['intact_KL','intact_KL_plus_native_response']:
  for domain,split,template in [('natural','evaluation',None),('expanded','evaluation','evaluation')]:
   chosen=[]
   for suffix in ['',':random',':coactivation']:
    matches=[x for x in r['summaries'] if x['seed']==seed and x['arm']==arm+suffix and x['domain']==domain and x['split']==split and x['template_split']==template and x['gate']=='all' and x['family'] is None]
    assert len(matches)==1;chosen.append(matches[0])
   scalar,random,learned=chosen
   rows.append(dict(seed=seed,source_arm=arm,domain=domain,split=split,template_split=template,endpoints=scalar['endpoints'],scalar_KL=scalar['mean_KL'],random_KL=random['mean_KL'],learned_KL=learned['mean_KL'],learned_random_ratio=learned['mean_KL']/random['mean_KL'],scalar_active=scalar['active_mean'],random_active=random['active_mean'],learned_active=learned['active_mean']))
out=dict(result_sha256=sha(rp),audit_sha256=sha(V/'coactivation_outputs_audit.json'),comparisons=rows,scope='Prespecified natural evaluation and jointly lexical/template-excluded development strata; all competence cohorts retained. Full other strata/families available in source results; not pristine holdouts.')
(V/'coactivation_output_comparison.json').write_text(json.dumps(out,indent=2)+'\n')
for x in rows:print(x)

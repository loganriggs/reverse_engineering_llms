from pathlib import Path
import json,math
import numpy as np
A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition');S='bilin18_copy_source_query_localization';j=json.loads((A/f'{S}.json').read_text());rows=[];rng=np.random.default_rng(790);rms=lambda x:math.sqrt(sum(v*v for v in x)/len(x));idx={(r['id'],r['receiver'],r['donor'],r['layer'],r['site']):r for r in j['records']};contrasts=[]
for fam in sorted({r['family'] for r in j['records']}):
 for mode in ['task','nuisance']:
  for l in [1,4]:
   for site in j['config']['sites']:
    rr=[r for r in j['records'] if r['family']==fam and r['mode']==mode and r['layer']==l and r['site']==site];row={'family':fam,'mode':mode,'layer':l,'site':site,'edits':len(rr),'margin_RMS':rms([r['margin_change'] for r in rr]),'patch_norm_RMS':math.sqrt(sum(r['patch_squared_norm'] for r in rr)/len(rr))}
    if mode=='task':
     vals=np.array([np.mean([r['donor_directed_change'] for r in rr if r['item']==item]) for item in sorted({r['item'] for r in rr})]);boot=vals[rng.integers(len(vals),size=(4096,len(vals)))].mean(1);row.update(donor_directed_mean=float(vals.mean()),descriptive_item_bootstrap_95=np.quantile(boot,[.025,.975]).tolist(),donor_choice_rate=sum(r['donor_choice'] for r in rr)/len(rr))
    rows.append(row)
   rr=[r for r in j['records'] if r['family']==fam and r['mode']==mode and r['layer']==l and r['site']=='both_sources'];gap=[];jointgap=[];means=[]
   for r in rr:
    k=r['id'],r['receiver'],r['donor'],l;gap.append(r['margin_change']-idx[*k,'first_source']['margin_change']-idx[*k,'second_source']['margin_change']);jointgap.append(idx[*k,'sources_and_query']['margin_change']-r['margin_change']-idx[*k,'query']['margin_change']);means.append((r['margin_change']-idx[*k,'query']['margin_change'])*(1 if r['receiver']==1 else -1))
   contrasts.append({'family':fam,'mode':mode,'layer':l,'both_sources_minus_sum_individuals_margin_RMS':rms(gap),'sources_query_minus_sum_margin_RMS':rms(jointgap),'both_sources_minus_query_directed_mean':sum(means)/len(means) if mode=='task' else None})
(A/f'{S}_comparisons.json').write_text(json.dumps({'summary':rows,'contrasts':contrasts,'limitations':'Descriptive bootstrap over lexical item clusters; no multiplicity correction. Patch norms and number of replaced tokens differ. Panel reused for localization.'},indent=2)+'\n');print('AUDITS',j['audits']);print('EXCLUDED',j['excluded'])
for r in rows:print({k:round(v,5) if isinstance(v,float) else v for k,v in r.items()})
for r in contrasts:print(r)

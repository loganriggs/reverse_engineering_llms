from pathlib import Path
import json,math
import numpy as np
A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition');S='bilin18_bilinear_odd_even_controls';j=json.loads((A/f'{S}.json').read_text());idx={(r['id'],r['receiver'],r['donor'],r['layer'],r['control'],r['term']):r['margin_change'] for r in j['records']};paired=[]
for r in j['records']:
 if r['term']!='plus':continue
 k=r['id'],r['receiver'],r['donor'],r['layer'],r['control'];plus=r['margin_change'];minus=idx[*k,'minus'];odd=(plus-minus)/2;even=(plus+minus)/2;paired.append({**{kk:r[kk] for kk in ['id','receiver','donor','layer','control','family','mode']},'paired_odd':odd,'paired_even':even,'odd_patch':idx[*k,'odd'],'even_patch':idx[*k,'even'],'donor_directed_odd':odd*(1 if r['receiver']==1 else -1) if r['mode']=='task' else None})
summary=[];rms=lambda x:math.sqrt(sum(v*v for v in x)/len(x))
for mode in ['task','nuisance']:
 for fam in sorted({r['family'] for r in paired}):
  for l in [1,4]:
   for ctrl in ['true','shuffle_next','random_pooled']:
    cc=[ctrl] if ctrl!='random_pooled' else ['random_7861','random_7862','random_7863'];rr=[r for r in paired if r['mode']==mode and r['family']==fam and r['layer']==l and r['control'] in cc];gg=[g for g in j['geometry'] if g['mode']==mode and g['family']==fam and g['layer']==l and g['control'] in cc and g['rho']==.3];summary.append({'mode':mode,'family':fam,'layer':l,'control':ctrl,'module_even_over_odd_norm':math.sqrt(sum(g['even_squared_norm'] for g in gg)/sum(g['odd_squared_norm'] for g in gg)),'paired_odd_RMS':rms([r['paired_odd'] for r in rr]),'paired_even_RMS':rms([r['paired_even'] for r in rr]),'odd_patch_vs_paired_odd_RMS':rms([r['odd_patch']-r['paired_odd'] for r in rr]),'directed_odd_mean':sum(r['donor_directed_odd'] for r in rr)/len(rr) if mode=='task' else None})
contrasts=[];rng=np.random.default_rng(787)
for fam in sorted({r['family'] for r in paired}):
 for l in [1,4]:
  for control in ['shuffle_next','random_pooled']:
   vals=[]
   for rid in sorted({r['id'] for r in paired if r['family']==fam}):
    rr=[r for r in paired if r['id']==rid and r['layer']==l and r['mode']=='task'];a=[r['donor_directed_odd'] for r in rr if r['control']=='true'];b=[r['donor_directed_odd'] for r in rr if r['control']==control or (control=='random_pooled' and r['control'].startswith('random'))];vals.append(np.mean(a)-np.mean(b))
   v=np.array(vals);boot=v[rng.integers(len(v),size=(4096,len(v)))].mean(1);contrasts.append({'family':fam,'layer':l,'contrast':'true_minus_'+control,'mean':float(v.mean()),'descriptive_item_bootstrap_95':np.quantile(boot,[.025,.975]).tolist(),'items':len(v)})
(A/f'{S}_comparisons.json').write_text(json.dumps({'summary':summary,'paired':paired,'contrasts':contrasts,'limitations':'Bootstrap clusters reciprocal edits and seeds by item; only eight reused items per family, no multiplicity correction; descriptive uncertainty, not confirmatory inference.'},indent=2)+'\n');print('AUDIT',j['audits'])
for fam in ['agreement','capital_recall','entity_binding','sequence_copy']:
 print(fam)
 for r in summary:
  if r['mode']=='task' and r['family']==fam:print({k:round(v,5) if isinstance(v,float) else v for k,v in r.items()})
 for r in contrasts:
  if r['family']==fam:print(r)

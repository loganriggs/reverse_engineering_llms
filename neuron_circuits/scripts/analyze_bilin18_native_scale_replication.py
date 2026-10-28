from pathlib import Path
import json,math
import numpy as np
A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition');S='bilin18_native_scale_replication';j=json.loads((A/f'{S}.json').read_text());idx={(r['id'],r['receiver'],r['donor'],r['layer'],r['control'],r['fraction'],r['sign']):r['margin_change'] for r in j['records']};paired=[]
for r in j['records']:
 if r['sign']!=1 or r['control'] in ['native_input','native_output','identity']:continue
 k=r['id'],r['receiver'],r['donor'],r['layer'],r['control'],r['fraction'];minus=idx[*k,-1];plus=r['margin_change'];sign=1 if r['receiver']==1 else -1;paired.append({**{kk:r[kk] for kk in ['id','receiver','donor','layer','control','fraction','family','mode']},'odd':(plus-minus)/2,'even':(plus+minus)/2,'plus':plus,'directed_odd':(plus-minus)/2*sign,'native':idx[r['id'],r['receiver'],r['donor'],r['layer'],'native_output',1.,1],'orientation':sign})
summary=[];contrasts=[];rng=np.random.default_rng(788);rms=lambda a:math.sqrt(sum(v*v for v in a)/len(a))
for mode in ['task','nuisance']:
 for fam in sorted({r['family'] for r in paired}):
  for l in [1,4]:
   for frac in [.25,.5,1.]:
    for ctrl in ['true','shuffle_next','random_pooled']:
     cc=[ctrl] if ctrl!='random_pooled' else ['random_7861','random_7862','random_7863'];rr=[r for r in paired if r['mode']==mode and r['family']==fam and r['layer']==l and r['fraction']==frac and r['control'] in cc];gg=[g for g in j['geometry'] if g['mode']==mode and g['family']==fam and g['layer']==l and g['fraction']==frac and g['sign']==1 and g['control'] in cc];summary.append({'mode':mode,'family':fam,'layer':l,'fraction':frac,'control':ctrl,'odd_RMS':rms([r['odd'] for r in rr]),'even_RMS':rms([r['even'] for r in rr]),'plus_RMS':rms([r['plus'] for r in rr]),'native_RMS':rms([r['native'] for r in rr]),'directed_odd_mean':sum(r['directed_odd'] for r in rr)/len(rr) if mode=='task' else None,'native_directed_mean':sum(r['native']*r['orientation'] for r in rr)/len(rr) if mode=='task' else None,'first_order_module_relative_error':math.sqrt(sum(g['interaction_squared_norm'] for g in gg)/sum(g['full_squared_norm'] for g in gg))})
    if mode!='task':continue
    for ctrl in ['zero','shuffle_next','random_pooled']:
     vals=[]
     for rid in sorted({r['id'] for r in paired if r['family']==fam}):
      rr=[r for r in paired if r['id']==rid and r['mode']=='task' and r['layer']==l and r['fraction']==frac];a=[r['directed_odd'] for r in rr if r['control']=='true'];b=[0.] if ctrl=='zero' else [r['directed_odd'] for r in rr if r['control']==ctrl or (ctrl=='random_pooled' and r['control'].startswith('random'))];vals.append(np.mean(a)-np.mean(b))
     v=np.array(vals);boot=v[rng.integers(len(v),size=(4096,len(v)))].mean(1);contrasts.append({'family':fam,'layer':l,'fraction':frac,'contrast':'true_minus_'+ctrl,'mean':float(v.mean()),'descriptive_item_bootstrap_95':np.quantile(boot,[.025,.975]).tolist()})
(A/f'{S}_comparisons.json').write_text(json.dumps({'summary':summary,'paired':paired,'contrasts':contrasts,'limitations':'Sixteen items per task-format cell, vocabulary shared across formats; item bootstrap groups reciprocal edits/seeds, descriptive without multiplicity adjustment.'},indent=2)+'\n');print(j['audits'])
for fam in ['entity_binding_f0','entity_binding_f1','sequence_copy_f0','sequence_copy_f1']:
 print(fam)
 for r in summary:
  if r['mode']=='task' and r['family']==fam and r['fraction']==1.:print({k:round(v,6) if isinstance(v,float) else v for k,v in r.items()})
 for r in contrasts:
  if r['family']==fam and r['fraction']==1.:print(r)

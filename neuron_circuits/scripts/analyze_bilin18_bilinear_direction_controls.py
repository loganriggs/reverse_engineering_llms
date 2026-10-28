from pathlib import Path
import json,math
A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition');S='bilin18_bilinear_direction_controls';j=json.loads((A/f'{S}.json').read_text());idx={(r['id'],r['receiver'],r['donor'],r['layer'],r['control'],r['term']):r['margin_change'] for r in j['records']};rows=[]
for mode in ['task','nuisance']:
 for fam in sorted({g['family'] for g in j['geometry']}):
  for rho in j['config']['radii']:
   for l in [1,4]:
    for ctrl in j['config']['controls']+['random_pooled']:
     controls=[ctrl] if ctrl!='random_pooled' else [c for c in j['config']['controls'] if c.startswith('random')];gg=[g for g in j['geometry'] if g['family']==fam and g['mode']==mode and g['layer']==l and g['rho']==rho and g['control'] in controls];den=sum(g['full_squared_norm'] for g in gg);row={'mode':mode,'family':fam,'rho':rho,'layer':l,'control':ctrl,'module_first_order_relative_error':math.sqrt(sum(g['interaction_squared_norm'] for g in gg)/den),'full_output_vector_RMS':math.sqrt(den/len(gg))}
     if rho==.3:
      rr=[r for r in j['records'] if r['family']==fam and r['mode']==mode and r['layer']==l and r['control'] in controls and r['term']=='full'];rms=lambda a:math.sqrt(sum(v*v for v in a)/len(a));row.update(full_margin_RMS=rms([r['margin_change'] for r in rr]),linear_margin_error_RMS=rms([r['margin_change']-idx[r['id'],r['receiver'],r['donor'],l,r['control'],'linear'] for r in rr]),donor_directed_mean=sum(r['margin_change']*(1 if r['receiver']==1 else -1) for r in rr)/len(rr) if mode=='task' else None)
     rows.append(row)
(A/f'{S}_comparisons.json').write_text(json.dumps({'rows':rows,'note':'Random-pooled rows combine three seeds at each same base; these are not independent task examples.'},indent=2)+'\n');print(j['audits'])
for mode in ['task','nuisance']:
 print(mode,'rho .3: tuple = control, layer, module error, full output RMS, margin RMS, approximation margin error, donor mean')
 for fam in sorted({r['family'] for r in rows}):
  print(fam,[(r['control'],r['layer'],*[round(r[k],4) if r.get(k) is not None else None for k in ['module_first_order_relative_error','full_output_vector_RMS','full_margin_RMS','linear_margin_error_RMS','donor_directed_mean']]) for r in rows if r['family']==fam and r['mode']==mode and r['rho']==.3 and r['control'] in ['true','shuffle_next','random_pooled']])
for mode in ['task','nuisance']:
 for rho in [.05,.15,.3]:
  for ctrl in ['true','shuffle_next','random_7861','random_7862','random_7863']:
   ii={(r['family'],r['layer']):r for r in rows if r['mode']==mode and r['rho']==rho and r['control']==ctrl};print('L1 larger',mode,rho,ctrl,sum(ii[f,1]['module_first_order_relative_error']>ii[f,4]['module_first_order_relative_error'] for f in sorted({r['family'] for r in rows})))

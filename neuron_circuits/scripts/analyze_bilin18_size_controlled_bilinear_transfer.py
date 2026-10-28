from pathlib import Path
import json,math
A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition');S='bilin18_size_controlled_bilinear_transfer';j=json.loads((A/f'{S}.json').read_text());idx={(r['id'],r['receiver'],r['donor'],r['layer'],r['rho'],r['term']):r for r in j['records']};rows=[]
for mode in ['task','nuisance']:
 for fam in sorted({r['family'] for r in j['records']}):
  for rho in j['config']['radii']:
   for l in [1,4]:
    gg=[g for g in j['geometry'] if g['mode']==mode and g['family']==fam and g['rho']==rho and g['layer']==l];rr=[r for r in j['records'] if r['mode']==mode and r['family']==fam and r['rho']==rho and r['layer']==l and r['term']=='full'];rms=lambda a: math.sqrt(sum(v*v for v in a)/len(a));den=sum(g['full_squared_norm'] for g in gg);rows.append({'family':fam,'mode':mode,'rho':rho,'layer':l,'n':len(rr),'module_first_order_relative_error':math.sqrt(sum(g['interaction_squared_norm'] for g in gg)/den) if den else None,'full_margin_RMS':rms([r['margin_change'] for r in rr]),'linear_margin_error_RMS':rms([r['margin_change']-idx[r['id'],r['receiver'],r['donor'],l,rho,'linear']['margin_change'] for r in rr])})
(A/f'{S}_comparisons.json').write_text(json.dumps({'rows':rows},indent=2)+'\n')
print('competence',j['competence']);print('audits',j['audits']);print('excluded',j['excluded'],'skipped',len(j['skipped']))
for mode in ['task','nuisance']:
 print(mode)
 for fam in sorted({r['family'] for r in rows}):
  rr=[r for r in rows if r['mode']==mode and r['family']==fam and r['rho']==.3];print(fam,[(r['layer'],round(r['module_first_order_relative_error'],4),round(r['full_margin_RMS'],4),round(r['linear_margin_error_RMS'],4)) for r in rr])
for mode in ['task','nuisance']:
 for rho in [.05,.15,.3]:
  count=0
  for fam in sorted({r['family'] for r in rows}):
   rr=[r for r in rows if r['mode']==mode and r['family']==fam and r['rho']==rho];count+=rr[0]['module_first_order_relative_error']>rr[1]['module_first_order_relative_error']
  print('L1 larger',mode,rho,count,'/8')

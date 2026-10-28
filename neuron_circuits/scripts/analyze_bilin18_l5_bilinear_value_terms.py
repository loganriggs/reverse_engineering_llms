from pathlib import Path
import json, math
A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition');stem='bilin18_l5_bilinear_value_terms';j=json.loads((A/f'{stem}.json').read_text());old=json.loads((A/'bilin18_l5_upstream_value_paths.json').read_text());idx={(r['id'],r['receiver'],r['donor'],r['site']):r['margin_change'] for r in j['records']};oi={(r['id'],r['receiver'],r['donor'],r['site']):r['margin_change'] for r in old['records']};rows=[];checks={'native_full_vs_H781_max_abs':0.,'analytic_full_vs_native_full_margin_max_abs':0.}
for r in j['records']:
 k=(r['id'],r['receiver'],r['donor']);site=r['site']
 if site.endswith('_native_full'):checks['native_full_vs_H781_max_abs']=max(checks['native_full_vs_H781_max_abs'],abs(r['margin_change']-oi[*k,site[:2]+'_mlp']))
 if site in ['L1_full','L4_full']:checks['analytic_full_vs_native_full_margin_max_abs']=max(checks['analytic_full_vs_native_full_margin_max_abs'],abs(r['margin_change']-idx[*k,site[:2]+'_native_full']))
for mode in ['task','nuisance']:
 for l in [1,4]:
  for term in ['left','right','interaction','linear','full','native_full']:
   rr=[r for r in j['records'] if r['mode']==mode and r['site']==f'L{l}_{term}'];rms=lambda v: math.sqrt(sum(x*x for x in v)/len(v));err=[r['margin_change']-idx[r['id'],r['receiver'],r['donor'],f'L{l}_native_full'] for r in rr];gg=[g for g in j['term_geometry'] if g['mode']==mode and g['layer']==l and g['term']==term];rows.append({'mode':mode,'layer':l,'term':term,'readout_margin_RMS':rms([r['margin_change'] for r in rr]),'margin_error_to_native_site_swap_RMS':rms(err),'module_delta_norm_relative_to_full':math.sqrt(sum(g['squared_norm'] for g in gg)/sum(g['target_squared_norm'] for g in gg))})
norm=[]
for mode in ['task','nuisance']:
 for l in [1,4]:
  gg=[g for g in j['normalization_geometry'] if g['mode']==mode and g['layer']==l];den=sum(g['delta_squared_norm'] for g in gg);norm.append({'mode':mode,'layer':l,'direction_relative_norm':math.sqrt(sum(g['direction_squared_norm'] for g in gg)/den),'scale_relative_norm':math.sqrt(sum(g['scale_squared_norm'] for g in gg)/den),'direction_scale_dot_over_delta_norm_squared':sum(g['direction_scale_dot'] for g in gg)/den})
assert checks['native_full_vs_H781_max_abs']<1e-4 and checks['analytic_full_vs_native_full_margin_max_abs']<1e-4
result={'summary':rows,'normalization':norm,'additional_audits':checks};(A/f'{stem}_comparisons.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));print(j['audits'])

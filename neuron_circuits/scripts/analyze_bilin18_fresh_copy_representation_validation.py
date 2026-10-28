from pathlib import Path
import json,numpy as np
A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition');S='bilin18_fresh_copy_representation_validation';j=json.loads((A/f'{S}.json').read_text());key=lambda r:(r['task'],r['id'],r['amp'],r['state'],r['arm'],r['dictionary']);replay=j['audits']['full_response_historical_replay_max_abs'];rng=np.random.default_rng(817);summary=[];contrasts=[]
for task in ['copy']:
 for group in ['all',0,1]:
  rr=[r for r in j['records'] if r['task']==task and (group=='all' or r['selected_group']==group)];zero=np.mean([r['KL_full_to_candidate'] for r in rr if r['dictionary']=='zero'])
  for name in ['zero','sharedprivate','shared_only','private_only','wrong_private']:
   x=[r for r in rr if r['dictionary']==name];summary.append({'task':task,'group':group,'dictionary':name,'variants':len(x),'items':len({r['item'] for r in x}),'mean_KL':float(np.mean([r['KL_full_to_candidate'] for r in x])),'KL_relative_to_zero':float(np.mean([r['KL_full_to_candidate'] for r in x])/zero),'mean_reverse_KL':float(np.mean([r['KL_candidate_to_full'] for r in x])),'mean_TV':float(np.mean([r['total_variation'] for r in x]))})
  chosen={key(r)[:-1]:r for r in rr if r['dictionary']=='sharedprivate'};other={key(r)[:-1]:r for r in rr if r['dictionary']=='wrong_private'};items=sorted({r['item'] for r in rr});diff=np.array([np.mean([other[k]['KL_full_to_candidate']-v['KL_full_to_candidate'] for k,v in chosen.items() if v['item']==i]) for i in items]);b=diff[rng.integers(len(diff),size=(4096,len(diff)))].mean(1);contrasts.append({'task':task,'group':group,'item_mean_other_minus_selected_KL':float(diff.mean()),'descriptive_item_bootstrap_CI':np.quantile(b,[.025,.975]).tolist(),'other_wins_variant_fraction':float(np.mean([other[k]['KL_full_to_candidate']<v['KL_full_to_candidate'] for k,v in chosen.items()]))})
(A/f'{S}_comparisons.json').write_text(json.dumps({'summary':summary,'contrasts':contrasts,'fresh_capture_full_response_replay_max_abs':replay,'limitations':'32 fresh copying items, no dictionary refit. Means pool correlated dose/arm/state variants; bootstrap resamples item means. KL relative-to-zero is ratio of mean KL, not RMS error or causal share. No gate changed from heldout comparison.'},indent=2)+'\n');print(j['audits']);print('fresh replay',replay)
for r in summary:print(r)
for r in contrasts:print(r)

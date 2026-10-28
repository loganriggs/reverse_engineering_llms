from pathlib import Path
import json,numpy as np
A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition');S='bilin18_fresh_copy_flat_controls';rng=np.random.default_rng(821);summary=[];contrasts=[]
for task,stem in [('copy',S),('ioi','bilin18_fresh_ioi_representation_validation')]:
 j=json.loads((A/f'{stem}.json').read_text());rr=j['records'];key=lambda r:(r['id'],r['amp'],r['state'],r['arm']);ref={key(r):r for r in rr if r['dictionary']=='sharedprivate'};zero=np.mean([r['KL_full_to_candidate'] for r in rr if r['dictionary']=='zero'])
 for name in ['zero','sharedprivate','flat5','flat8']:
  x=[r for r in rr if r['dictionary']==name];summary.append({'task':task,'dictionary':name,'items':len({r['item'] for r in x}),'mean_KL':float(np.mean([r['KL_full_to_candidate'] for r in x])),'KL_relative_to_zero':float(np.mean([r['KL_full_to_candidate'] for r in x])/zero),'mean_TV':float(np.mean([r['total_variation'] for r in x])),'accuracy':float(np.mean([r['correct'] for r in x]))})
  if name not in ['flat5','flat8']:continue
  diffs=np.array([np.mean([r['KL_full_to_candidate']-ref[key(r)]['KL_full_to_candidate'] for r in x if r['item']==i]) for i in sorted({r['item'] for r in x})]);b=diffs[rng.integers(len(diffs),size=(4096,len(diffs)))].mean(1);contrasts.append({'task':task,'comparison':name+' minus sharedprivate KL','item_mean':float(diffs.mean()),'CI':np.quantile(b,[.025,.975]).tolist(),'items_flat_better':int((diffs<0).sum()),'items':len(diffs)})
(A/f'{S}_comparisons.json').write_text(json.dumps({'summary':summary,'contrasts':contrasts,'limitations':'Fresh lexical panels, frozen fits. Shared/private stores8 activates5; flat5 stores5 activates5; flat8 stores8 activates8. Decoder column count is not a complete compute/encoder cost measure. Bootstrap lexical items, descriptive.'},indent=2)+'\n')
for r in summary+contrasts:print(r)

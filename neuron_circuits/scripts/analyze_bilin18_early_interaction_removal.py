from pathlib import Path
import json,math
import numpy as np
A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition');S='bilin18_early_interaction_removal';j=json.loads((A/f'{S}.json').read_text());idx={(r['id'],r['a'],r['b'],r['arm']):r for r in j['records']};rows=[]
def vec(r):return np.array(r['candidate_logit_changes'])
def inter(r):return vec(r)-vec(idx[r['id'],r['a'],0,r['arm']])-vec(idx[r['id'],0,r['b'],r['arm']])+vec(idx[r['id'],0,0,r['arm']])
for r in j['records']:
 if r['a']==0 or r['b']==0:continue
 base=idx[r['id'],r['a'],r['b'],'none'];receiver=r['receiver_index'];targets=sorted({r['variant_indices'][r['a']],r['variant_indices'][r['b']]});other=[i for i in range(16) if i not in targets+[receiver]];con=lambda v:float(v[targets].mean()-v[other].mean());I=inter(r);B=inter(base);C=I-I.mean();BC=B-B.mean();rows.append({'id':r['id'],'item':r['item'],'format':r['format'],'kind':'coherent' if r['a']==r['b'] else 'mixed','arm':r['arm'],'target_specificity':con(vec(r)),'specificity_change':con(vec(r)-vec(base)),'remaining_interaction_specificity':con(I),'baseline_interaction_specificity':con(B),'centered_interaction_squared_norm':float(C@C),'baseline_centered_interaction_squared_norm':float(BC@BC),'candidate_effect_change_squared_norm':float(np.square((vec(r)-vec(base))-(vec(r)-vec(base)).mean()).sum())})
summary=[];rng=np.random.default_rng(794)
for f in [0,1]:
 for kind in ['coherent','mixed']:
  for arm in j['config']['arms']:
   rr=[r for r in rows if r['format']==f and r['kind']==kind and r['arm']==arm];diff=np.array([np.mean([r['specificity_change'] for r in rr if r['item']==i]) for i in range(16)]);boot=diff[rng.integers(16,size=(4096,16))].mean(1);summary.append({'format':f,'kind':kind,'arm':arm,'target_specificity_mean':float(np.mean([r['target_specificity'] for r in rr])),'specificity_change_mean':float(diff.mean()),'specificity_change_descriptive_item_bootstrap_95':np.quantile(boot,[.025,.975]).tolist(),'remaining_centered_interaction_RMS_over_baseline':math.sqrt(sum(r['centered_interaction_squared_norm'] for r in rr)/sum(r['baseline_centered_interaction_squared_norm'] for r in rr)),'remaining_specificity_interaction_RMS':math.sqrt(np.mean([r['remaining_interaction_specificity']**2 for r in rr])),'baseline_specificity_interaction_RMS':math.sqrt(np.mean([r['baseline_interaction_specificity']**2 for r in rr]))})
(A/f'{S}_comparisons.json').write_text(json.dumps({'summary':summary,'rows':rows,'limitations':'Remaining-interaction norm ratios are intervention comparisons, not unique causal shares. Sixteen reused item clusters/format; descriptive bootstrap without multiplicity correction.'},indent=2)+'\n');print(j['audits']);print(json.dumps(summary,indent=2))

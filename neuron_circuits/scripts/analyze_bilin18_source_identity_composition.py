from pathlib import Path
import json,math
import numpy as np
A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition');S='bilin18_source_identity_composition';j=json.loads((A/f'{S}.json').read_text());idx={(r['id'],r['a'],r['b']):r for r in j['records']};out=[];balanced=[]
for r in j['records']:
 a,b=r['a'],r['b'];rid=r['id']
 if a==0 or b==0:continue
 y=np.array(r['candidate_logit_changes']);ya=np.array(idx[rid,a,0]['candidate_logit_changes']);yb=np.array(idx[rid,0,b]['candidate_logit_changes']);z=np.array(idx[rid,0,0]['candidate_logit_changes']);inter=y-ya-yb+z;receiver=r['receiver_index'];targets=sorted({r['variant_indices'][a],r['variant_indices'][b]});others=[k for k in range(len(y)) if k not in targets+[receiver]];contrast=lambda x:float(x[targets].mean()-x[others].mean());out.append({'id':rid,'item':r['item'],'format':r['format'],'kind':'coherent' if a==b else 'mixed','a':a,'b':b,'target_specificity':contrast(y),'additive_target_specificity':contrast(ya+yb-z),'interaction_specificity':contrast(inter),'interaction_centered_squared_norm':float(np.square(inter-inter.mean()).sum()),'actual_centered_squared_norm':float(np.square(y-y.mean()).sum()),'receiver_logprob_change':r['candidate_logprob_changes'][receiver],'target_logprob_change':float(np.mean(np.array(r['candidate_logprob_changes'])[targets]))})
for rid in sorted({r['id'] for r in j['records']}):
 r=idx[rid,0,0];receiver=r['receiver_index']
 for a,b in [(1,2),(1,3),(2,3)]:
  aa=idx[rid,a,a];bb=idx[rid,b,b];ab=idx[rid,a,b];ba=idx[rid,b,a];D=(np.array(aa['candidate_logit_changes'])+np.array(bb['candidate_logit_changes'])-np.array(ab['candidate_logit_changes'])-np.array(ba['candidate_logit_changes']))/2;targets=[r['variant_indices'][a],r['variant_indices'][b]];other=[k for k in range(len(D)) if k not in targets+[receiver]];energy=(aa['patch_squared_norm']+bb['patch_squared_norm']-ab['patch_squared_norm']-ba['patch_squared_norm'])/max(aa['patch_squared_norm']+bb['patch_squared_norm'],1e-12);balanced.append({'id':rid,'item':r['item'],'format':r['format'],'a':a,'b':b,'coherent_minus_mixed_target_specificity':float(D[targets].mean()-D[other].mean()),'coherent_minus_mixed_target_receiver_margin':float(D[targets].mean()-D[receiver]),'balanced_patch_energy_relative_difference':energy})
rng=np.random.default_rng(792);summary=[]
def cluster(rows,key):
 v=np.array([np.mean([r[key] for r in rows if r['item']==i]) for i in sorted({r['item'] for r in rows})]);boot=v[rng.integers(len(v),size=(4096,len(v)))].mean(1);return {'mean':float(v.mean()),'descriptive_item_bootstrap_95':np.quantile(boot,[.025,.975]).tolist()}
for f in [0,1]:
 for kind in ['coherent','mixed']:
  rr=[r for r in out if r['format']==f and r['kind']==kind];s={'format':f,'kind':kind}
  for m in ['target_specificity','additive_target_specificity','interaction_specificity','receiver_logprob_change','target_logprob_change']:s[m]=cluster(rr,m)
  s['centered_candidate_additive_relative_RMS_error']=math.sqrt(sum(r['interaction_centered_squared_norm'] for r in rr)/sum(r['actual_centered_squared_norm'] for r in rr));summary.append(s)
 rr=[r for r in balanced if r['format']==f];summary.append({'format':f,'kind':'balanced_coherent_minus_mixed','target_specificity':cluster(rr,'coherent_minus_mixed_target_specificity'),'target_receiver_margin':cluster(rr,'coherent_minus_mixed_target_receiver_margin'),'max_abs_balanced_patch_energy_relative_difference':max(abs(r['balanced_patch_energy_relative_difference']) for r in rr)})
(A/f'{S}_comparisons.json').write_text(json.dumps({'summary':summary,'interactions':out,'balanced_contrasts':balanced,'limitations':'16 reused lexical-item clusters/format; donor pairs and formats correlated; no multiplicity correction. Balanced contrast cancels additive main effects and balances aggregate patch energy, but not all nonlinear geometry.'},indent=2)+'\n');print(j['audits']);print(json.dumps(summary,indent=2))

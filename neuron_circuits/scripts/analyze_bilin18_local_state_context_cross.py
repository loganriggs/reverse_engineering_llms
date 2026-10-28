from pathlib import Path
import json
import numpy as np
A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition');S='bilin18_local_state_context_cross';j=json.loads((A/f'{S}.json').read_text());rng=np.random.default_rng(800);summary=[];contrasts=[]
def boot(v):
 v=np.array(v);b=v[rng.integers(len(v),size=(4096,len(v)))].mean(1);return {'mean':float(v.mean()),'descriptive_item_bootstrap_95':np.quantile(b,[.025,.975]).tolist()}
for f in [0,1]:
 for delta in [.25,.5]:
  ix={}
  for c in [False,True]:
   for s in [False,True]:
    rr=[r for r in j['records'] if r['format']==f and r['delta']==delta and r['context_same_key']==c and r['state_same_key']==s];v=[]
    for i in sorted({r['item'] for r in rr}):
     x=np.mean([r['intended_logprob_response'] for r in rr if r['item']==i]);ix[i,c,s]=x;v.append(x)
    summary.append({'format':f,'delta':delta,'context_same_key':c,'state_same_key':s,'response':boot(v),'hybrid_baseline_logprob_change_mean':float(np.mean([r['hybrid_baseline_logprob_change_from_native'] for r in rr])),'hybrid_baseline_KL_mean':float(np.mean([r['KL_native_to_hybrid_baseline'] for r in rr])),'hybrid_baseline_accuracy':float(np.mean([r['hybrid_baseline_full_correct'] for r in rr])),'perturbed_accuracy':float(np.mean([r['perturbed_full_correct'] for r in rr]))})
  ids=sorted({i for i,c,s in ix})
  for name in ['context_effect_fixed_different_state','context_effect_fixed_same_state','state_effect_fixed_different_context','state_effect_fixed_same_context','average_context_effect','average_state_effect','interaction','native_diagonal_gap']:
   vals=[]
   for i in ids:
    a,b,c,d=ix[i,False,False],ix[i,False,True],ix[i,True,False],ix[i,True,True]
    vals.append({'context_effect_fixed_different_state':a-c,'context_effect_fixed_same_state':b-d,'state_effect_fixed_different_context':a-b,'state_effect_fixed_same_context':c-d,'average_context_effect':(a-c+b-d)/2,'average_state_effect':(a-b+c-d)/2,'interaction':d-c-b+a,'native_diagonal_gap':a-d}[name])
   contrasts.append({'format':f,'delta':delta,'contrast':name,**boot(vals)})
(A/f'{S}_comparisons.json').write_text(json.dumps({'summary':summary,'contrasts':contrasts,'limitations':'Controlled hybrid response differences; not unique causal shares. Average state/context contrasts sum to native diagonal response gap algebraically. Item bootstrap averages future queries, descriptive without multiplicity correction.'},indent=2)+'\n');print(j['audits'])
for r in summary:
 if r['delta']==.5:print(r)
for r in contrasts:
 if r['delta']==.5:print(r)

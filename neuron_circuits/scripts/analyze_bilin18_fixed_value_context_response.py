from pathlib import Path
import json
import numpy as np
A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition');S='bilin18_fixed_value_context_response';j=json.loads((A/f'{S}.json').read_text());rng=np.random.default_rng(799);summary=[];contrasts=[]
def interval(x):
 x=np.array(x);boot=x[rng.integers(len(x),size=(4096,len(x)))].mean(1);return {'mean':float(x.mean()),'descriptive_item_bootstrap_95':np.quantile(boot,[.025,.975]).tolist()}
for f in [0,1]:
 for delta in j['config']['scales']:
  for sk in [False,True]:
   rr=[r for r in j['records'] if r['format']==f and r['scale']==delta and r['same_key']==sk];x=[np.mean([r['intended_logprob_change'] for r in rr if r['item']==i]) for i in sorted({r['item'] for r in rr})];summary.append({'format':f,'coefficient_delta':delta,'same_key':sk,'intended_logprob_change':interval(x),'KL_mean':float(np.mean([r['KL_native_to_scaled'] for r in rr])),'full_accuracy':float(np.mean([r['scaled_full_correct'] for r in rr]))})
  for query in ['average',0,1]:
   rr=[r for r in j['records'] if r['format']==f and r['scale']==delta and (query=='average' or r['query']==query)];x=[]
   for i in sorted({r['item'] for r in rr}):
    a=np.mean([r['intended_logprob_change'] for r in rr if r['item']==i and not r['same_key']]);b=np.mean([r['intended_logprob_change'] for r in rr if r['item']==i and r['same_key']]);x.append(a-b)
   contrasts.append({'format':f,'coefficient_delta':delta,'query':query,'different_minus_same_key_response':interval(x)})
(A/f'{S}_comparisons.json').write_text(json.dumps({'summary':summary,'paired_key_contrasts':contrasts,'limitations':'Same injected direction/magnitude/position within each item/format, while computational background differs. Item bootstrap averages duplicate/future-query variants, descriptive without multiplicity adjustment.'},indent=2)+'\n');print(j['audits'])
for r in summary:print(r)
for r in contrasts:
 if r['query']=='average' or r['coefficient_delta']==.5:print(r)

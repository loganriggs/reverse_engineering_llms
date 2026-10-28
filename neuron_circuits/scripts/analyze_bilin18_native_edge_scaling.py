from pathlib import Path
import json
import numpy as np
A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition');S='bilin18_native_edge_scaling';j=json.loads((A/f'{S}.json').read_text());summary=[];rng=np.random.default_rng(798)
for f in [0,1]:
 for sk in [False,True]:
  for sv in [False,True]:
   for scale in j['config']['scales']:
    rr=[r for r in j['records'] if r['format']==f and r['same_key']==sk and r['same_value']==sv and r['scale']==scale];row={'format':f,'same_key':sk,'same_value':sv,'scale':scale,'contexts':len(rr),'KL_mean':float(np.mean([r['KL_native_to_scaled'] for r in rr])),'KL_max':max(r['KL_native_to_scaled'] for r in rr)}
    if rr[0]['expected_index'] is not None:
     x=np.array([np.mean([r['intended_logprob_change'] for r in rr if r['item']==i]) for i in sorted({r['item'] for r in rr})]);boot=x[rng.integers(len(x),size=(4096,len(x)))].mean(1);row.update(intended_logprob_change_mean=float(x.mean()),descriptive_item_bootstrap_95=np.quantile(boot,[.025,.975]).tolist(),native_intended_prob_mean=float(np.mean([r['native_intended_prob'] for r in rr])),scaled_intended_prob_mean=float(np.mean([r['scaled_intended_prob'] for r in rr])),native_full_accuracy=float(np.mean([r['native_full_correct'] for r in rr])),scaled_full_accuracy=float(np.mean([r['scaled_full_correct'] for r in rr])),scaled_candidate_accuracy=float(np.mean([r['scaled_candidate_correct'] for r in rr])))
    summary.append(row)
(A/f'{S}_comparisons.json').write_text(json.dumps({'summary':summary,'limitations':'Item clusters average future-query contexts, including duplicates under same keys; descriptive intervals without multiplicity adjustment. Poor-binding cells cannot support successful-task claims; conflict cells have no target accuracy.'},indent=2)+'\n');print(j['audits'])
for r in summary:
 if r['same_value'] or r['scale']==0:print(r)

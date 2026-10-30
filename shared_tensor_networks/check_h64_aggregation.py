import copy,json
from pathlib import Path
import numpy as np
from aggregate_h64 import aggregate

rows=[]
for j,seed in enumerate(range(256101,256106),1):
 for n in [8192,32768]:
  for t in [520,2080]:
   for init in ['257001','257002','teacher']:
    # Deliberately unequal starts and teacher performance expose pooling mistakes.
    value=(j if init=='257001' else 3*j) if init!='teacher' else .001
    value*= (1 if n==8192 else .1)*(1 if t==520 else .5)
    rows.append(dict(name=f'{seed}_{n}_{t}_{init}',teacher=seed,n=n,tasks=t,initialization=init,train_mse=.01 if init=='teacher' else .1,
                     population_mse=[value]*t,component_cosines=[1. if init=='teacher' else .98]*64,transfer={'learned_8192':{'errors':[value]*4}}))
result=aggregate(rows)
for c in result['conditions']:
 expected=(6 if c['initialization']=='random' else .001)*(1 if c['n']==8192 else .1)*(1 if c['tasks']==520 else .5)
 np.testing.assert_allclose(c['median_transfer'],expected,rtol=1e-14)
 assert c['recovery']==(5 if c['initialization']=='teacher' else 0)
assert result['equal_budget']['more_inputs_wins']==5 and all(v['more_task_wins']==5 for v in result['fixed_input'].values())
assert sum(w['lower_by_margin'] for w in result['lower_loss_witnesses'])==40
for altered in [rows[:-1],rows+[rows[0]],rows[:-1]+[rows[0]]]:
 try:aggregate(altered)
 except AssertionError:pass
 else:raise AssertionError('Missing/duplicate condition accepted')
bad=copy.deepcopy(rows);bad[0]['transfer']['learned_8192']['errors'][0]=float('nan')
try:aggregate(bad)
except AssertionError:pass
else:raise AssertionError('Nonfinite transfer accepted')
out=Path('results/h64_aggregation');out.mkdir(exist_ok=True)
(out/'checks.json').write_text(json.dumps(dict(expected_medians=True,teacher_starts_separate=True,paired_comparisons=True,witnesses=40,missing_duplicate_nonfinite_rejected=True),indent=2)+'\n');print('Aggregation fixtures passed; no scientific outcomes evaluated.')

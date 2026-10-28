from pathlib import Path
import json
import numpy as np
A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition');S='bilin18_native_key_value_cross';j=json.loads((A/f'{S}.json').read_text());rr=[r for r in j['records'] if r['query']==0];rng=np.random.default_rng(797);summary=[];contrasts=[]
for f in [0,1]:
 for sk in [False,True]:
  for sv in [False,True]:
   rows=[r for r in rr if r['format']==f and r['same_key']==sk and r['same_value']==sv];allrows=[r for r in j['records'] if r['format']==f and r['same_key']==sk and r['same_value']==sv];summary.append({'format':f,'same_key':sk,'same_value':sv,'items':len(rows),**{m:float(np.mean([r[m] for r in rows])) for m in ['pattern','score1','score2','input_cosine']},'candidate_accuracy':None if sk and not sv else float(np.mean([r['candidate_correct'] for r in allrows])),'full_vocab_top1':None if sk and not sv else float(np.mean([r['full_vocab_correct'] for r in allrows]))})
 ids=sorted({r['item'] for r in rr if r['format']==f});idx={(r['item'],r['same_key'],r['same_value']):r for r in rr if r['format']==f}
 for name in ['value_effect_different_keys','value_effect_same_keys','key_effect_different_values','key_effect_same_values','key_value_interaction']:
  vals=[]
  for i in ids:
   z=lambda k,v:idx[i,k,v]['pattern'];v=z(False,True)-z(False,False) if name=='value_effect_different_keys' else z(True,True)-z(True,False) if name=='value_effect_same_keys' else z(True,False)-z(False,False) if name=='key_effect_different_values' else z(True,True)-z(False,True) if name=='key_effect_same_values' else z(True,True)-z(True,False)-z(False,True)+z(False,False);vals.append(v)
  x=np.array(vals);boot=x[rng.integers(len(x),size=(4096,len(x)))].mean(1);contrasts.append({'format':f,'contrast':name,'mean':float(x.mean()),'descriptive_item_bootstrap_95':np.quantile(boot,[.025,.975]).tolist()})
# Descriptive prediction on unseen item indices; vocab overlap remains across item indices.
y=np.array([r['pattern'] for r in rr]);X={
 'constant_format':np.array([[1,r['format']] for r in rr],float),
 'cosine_linear':np.array([[1,r['format'],r['input_cosine']] for r in rr],float),
 'cosine_quadratic':np.array([[1,r['format'],r['input_cosine'],r['input_cosine']**2] for r in rr],float),
 'identity_factorial':np.array([[1,r['format'],r['same_key'],r['same_value'],r['same_key']*r['same_value']] for r in rr],float)};predictions=[]
for name,x in X.items():
 pred=np.zeros(len(rr))
 for item in sorted({r['item'] for r in rr}):
  mask=np.array([r['item']==item for r in rr]);beta=np.linalg.lstsq(x[~mask],y[~mask],rcond=None)[0];pred[mask]=x[mask]@beta
 predictions.append({'model':name,'leave_item_out_RMS_error':float(np.sqrt(np.mean((pred-y)**2))),'target_RMS':float(np.sqrt(np.mean(y*y)))})
(A/f'{S}_comparisons.json').write_text(json.dumps({'summary':summary,'contrasts':contrasts,'descriptive_predictors':predictions,'limitations':'Value identity and residual similarity covary; observational prediction cannot distinguish them causally. Leave-item-out is not leave-word-out; candidate words overlap across items. Query variants not independent routing observations.'},indent=2)+'\n');print(j['audits']);print('excluded',j['excluded']);print(json.dumps({'summary':summary,'contrasts':contrasts,'prediction':predictions},indent=2))

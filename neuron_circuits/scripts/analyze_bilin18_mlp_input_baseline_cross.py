from pathlib import Path
import json,numpy as np
A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition');S='bilin18_mlp_input_baseline_cross';j=json.loads((A/f'{S}.json').read_text());rng=np.random.default_rng(806);summary=[]
def boot(v):
 v=np.array(v);b=v[rng.integers(len(v),size=(4096,len(v)))].mean(1);return {'mean':float(v.mean()),'CI':np.quantile(b,[.025,.975]).tolist()}
for f in [0,1]:
 rr=[r for r in j['records'] if r['format']==f and r['delta']==.5];ids=sorted({r['item'] for r in rr});gap={}
 for arm in ['own','input_swap','baseline_swap','both']:
  x=[r for r in rr if r['arm']==arm];gap[arm]=np.array([np.mean([r['response'] for r in x if r['item']==i and not r['context_same_key']])-np.mean([r['response'] for r in x if r['item']==i and r['context_same_key']]) for i in ids]);summary.append({'format':f,'arm':arm,'gap':boot(gap[arm]),'change_from_own':boot(gap[arm]-gap['own']),'different_mean':float(np.mean([r['response'] for r in x if not r['context_same_key']])),'same_mean':float(np.mean([r['response'] for r in x if r['context_same_key']])),'accuracy':float(np.mean([r['correct'] for r in x]))})
 summary.append({'format':f,'contrast':'input_swap_minus_both','value':boot(gap['input_swap']-gap['both'])})
 summary.append({'format':f,'contrast':'factorial_interaction','value':boot(gap['both']-gap['input_swap']-gap['baseline_swap']+gap['own'])})
quad=[]
for f in [0,1]:
 for arm in ['own','input_swap','baseline_swap','both']:
  x=[r for r in j['decomposition'] if r['format']==f and r['arm']==arm];quad.append({'format':f,'arm':arm,'quadratic_response_RMS_ratio':float(np.sqrt(np.mean([r['quadratic_norm']**2 for r in x])/np.mean([r['response_norm']**2 for r in x])))})
(A/f'{S}_comparisons.json').write_text(json.dumps({'summary':summary,'quadratic':quad,'limitations':'Descriptive paired item bootstrap after averaging query and local L2 state. Raw baseline/response crossed before RMS; quadratic term only describes bilinear map in normalized coordinates, not raw-input nonlinearity or causal share.'},indent=2)+'\n');print(j['audits'])
for r in summary+quad:print(r)

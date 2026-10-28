from pathlib import Path
import json,numpy as np
A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition');S='bilin18_indirect_object_response_cross';j=json.loads((A/f'{S}.json').read_text());rng=np.random.default_rng(810);summary=[];contrast=[]
def boot(v):
 v=np.asarray(v);b=v[rng.integers(len(v),size=(4096,len(v)))].mean(1);return {'mean':float(v.mean()),'CI':np.quantile(b,[.025,.975]).tolist()}
for f in [0,1]:
 for amp in [.25,.5]:
  rr=[r for r in j['records'] if r['format']==f and r['delta']==amp];ids=sorted({r['item'] for r in rr})
  for metric in ['first_logprob_response','margin_response']:
   gaps={}
   for arm in ['own','input_swap','baseline_swap','both']:
    x=[r for r in rr if r['arm']==arm];gaps[arm]=np.array([np.mean([r[metric] for r in x if r['item']==i and r['first_name_correct']])-np.mean([r[metric] for r in x if r['item']==i and not r['first_name_correct']]) for i in ids]);summary.append({'format':f,'amplitude':amp,'metric':metric,'arm':arm,'gap':boot(gaps[arm]),'change_from_own':boot(gaps[arm]-gaps['own']),'first_correct_mean':float(np.mean([r[metric] for r in x if r['first_name_correct']])),'second_correct_mean':float(np.mean([r[metric] for r in x if not r['first_name_correct']])),'correct_logprob_mean':float(np.mean([r['correct_logprob_response'] for r in x])),'mean_KL':float(np.mean([r['KL'] for r in x])),'accuracy':float(np.mean([r['correct'] for r in x]))})
   contrast.append({'format':f,'amplitude':amp,'metric':metric,'input_minus_baseline_gap':boot(gaps['input_swap']-gaps['baseline_swap'])})
(A/f'{S}_comparisons.json').write_text(json.dumps({'summary':summary,'contrasts':contrast,'limitations':'Eight name pairs; order variants averaged within pair for descriptive bootstrap. Two surfaces correlated; no multiplicity correction. Fixed-token logprob can change from competition, so fixed first-minus-second logit margin is also reported. Hybrid response interventions need not be on-manifold.'},indent=2)+'\n');print(j['audits'])
for r in summary:
 if r['amplitude']==.5:print(r)
for r in contrast:print(r)

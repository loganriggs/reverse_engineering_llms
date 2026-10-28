from pathlib import Path
import json,numpy as np
A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition');S='bilin18_mlp_input_baseline_fresh';j=json.loads((A/f'{S}.json').read_text());native=json.loads((A/'bilin18_fresh_bird_capture.json').read_text());rng=np.random.default_rng(807);summary=[];competence=[]
def boot(v):
 v=np.array(v);b=v[rng.integers(len(v),size=(4096,len(v)))].mean(1);return {'mean':float(v.mean()),'CI':np.quantile(b,[.025,.975]).tolist()}
for f in [0,1]:
 n=[r for r in native['records'] if r['format']==f];items=len({r['item'] for r in n});cand=np.mean([r['candidate_correct'] for r in n]);full=np.mean([r['full_vocab_correct'] for r in n]);competence.append({'format':f,'items':items,'candidate_accuracy_vs_train':float(cand),'full_accuracy':float(full),'locked_gate_pass':bool(items>=8 and cand>=.75 and full>=.5)})
 for amp in [.25,.5]:
  rr=[r for r in j['records'] if r['format']==f and r['delta']==amp];ids=sorted({r['item'] for r in rr});gap={}
  for arm in ['own','input_swap','baseline_swap','both']:
   x=[r for r in rr if r['arm']==arm];gap[arm]=np.array([np.mean([r['response'] for r in x if r['item']==i and not r['context_same_key']])-np.mean([r['response'] for r in x if r['item']==i and r['context_same_key']]) for i in ids]);summary.append({'format':f,'amplitude':amp,'arm':arm,'gap':boot(gap[arm]),'change_from_own':boot(gap[arm]-gap['own']),'probability_change_different':float(np.mean([r['probability_change'] for r in x if not r['context_same_key']])),'probability_change_same':float(np.mean([r['probability_change'] for r in x if r['context_same_key']])),'mean_KL':float(np.mean([r['KL_base_to_output'] for r in x])),'accuracy':float(np.mean([r['correct'] for r in x]))})
  summary.append({'format':f,'amplitude':amp,'contrast':'input_swap_minus_baseline_swap','value':boot(gap['input_swap']-gap['baseline_swap'])})
quad=[]
for f in [0,1]:
 for amp in [.25,.5]:
  for arm in ['own','input_swap','baseline_swap','both']:
   x=[r for r in j['decomposition'] if r['format']==f and r['amplitude']==amp and r['arm']==arm];quad.append({'format':f,'amplitude':amp,'arm':arm,'quadratic_response_RMS_ratio':float(np.sqrt(np.mean([r['quadratic_norm']**2 for r in x])/np.mean([r['response_norm']**2 for r in x])))})
(A/f'{S}_comparisons.json').write_text(json.dumps({'competence':competence,'summary':summary,'quadratic':quad,'exclusions':native['excluded'],'limitations':'Primary .5 contrast frozen; .25 secondary. Gate failures are descriptive only. Item bootstrap over state/query averages, no multiplicity correction. Candidate foil train is weak; full-vocabulary competence reported.'},indent=2)+'\n');print(j['audits']);print(competence)
for r in summary:print(r)

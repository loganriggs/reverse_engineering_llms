from pathlib import Path
import json,numpy as np
A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition');S='bilin18_transition_module_transfer';j=json.loads((A/f'{S}.json').read_text());old=json.loads((A/'bilin18_local_state_context_cross.json').read_text());rng=np.random.default_rng(805)
def boot(v):
 v=np.array(v);b=v[rng.integers(len(v),size=(4096,len(v)))].mean(1);return {'mean':float(v.mean()),'CI':np.quantile(b,[.025,.975]).tolist()}
key=lambda r:(r['id'],r['state_same_key'],r['delta'])
ref={key(r):r for r in old['records']};none={(r['layer'],key(r)):r for r in j['records'] if r['arm']=='none'}
audit={'H800_none_response_max_abs':max(abs(r['intended_logprob_response']-ref[key(r)]['intended_logprob_response']) for r in none.values()),'baseline_logprob_max_abs_vs_none':max(abs(r['hybrid_baseline_logprob_change_from_native']-none[r['layer'],key(r)]['hybrid_baseline_logprob_change_from_native']) for r in j['records']),'baseline_KL_max_abs_vs_none':max(abs(r['KL_native_to_hybrid_baseline']-none[r['layer'],key(r)]['KL_native_to_hybrid_baseline']) for r in j['records'])}
assert max(audit.values())<1e-4
rows=[]
for f in [0,1]:
 for layer in [5,6]:
  rr=[r for r in j['records'] if r['format']==f and r['delta']==.5 and r['layer']==layer];items=sorted({r['item'] for r in rr});gaps={}
  for arm in ['none','attention','mlp','both']:
   x=[r for r in rr if r['arm']==arm];gaps[arm]=[]
   for i in items:
    cells={c:np.mean([r['intended_logprob_response'] for r in x if r['item']==i and r['context_same_key']==c]) for c in [False,True]};gaps[arm].append(cells[False]-cells[True])
   rows.append({'format':f,'layer':layer,'arm':arm,'gap':boot(gaps[arm]),'change':boot(np.array(gaps[arm])-gaps['none']),'different_mean':float(np.mean([r['intended_logprob_response'] for r in x if not r['context_same_key']])),'same_mean':float(np.mean([r['intended_logprob_response'] for r in x if r['context_same_key']])),'accuracy':float(np.mean([r['perturbed_full_correct'] for r in x]))})
(A/f'{S}_comparisons.json').write_text(json.dumps({'summary':rows,'audits':audit,'limitations':'Item bootstrap over query/state averages. Descriptive reused panel, no multiplicity correction; controlled module responses, joint arm uses original MLP baseline/response.'},indent=2)+'\n');print(audit)
for r in rows:print(r)

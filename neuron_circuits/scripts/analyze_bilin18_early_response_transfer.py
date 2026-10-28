from pathlib import Path
import json,numpy as np
A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition');S='bilin18_early_response_transfer';j=json.loads((A/f'{S}.json').read_text());old=json.loads((A/'bilin18_local_state_context_cross.json').read_text());rng=np.random.default_rng(802)
def boot(v):
 v=np.array(v);b=v[rng.integers(len(v),size=(4096,len(v)))].mean(1);return {'mean':float(v.mean()),'descriptive_item_bootstrap_95':np.quantile(b,[.025,.975]).tolist()}
key=lambda r:(r['id'],r['state_same_key'],r['delta'])
ref={key(r):r for r in old['records']};none={key(r):r for r in j['records'] if r['arm']=='none'}
audit={'H800_none_response_max_abs':max(abs(r['intended_logprob_response']-ref[key(r)]['intended_logprob_response']) for r in none.values()),'zero_delta_baseline_logprob_max_abs_vs_none':max(abs(r['hybrid_baseline_logprob_change_from_native']-none[key(r)]['hybrid_baseline_logprob_change_from_native']) for r in j['records']),'zero_delta_baseline_KL_max_abs_vs_none':max(abs(r['KL_native_to_hybrid_baseline']-none[key(r)]['KL_native_to_hybrid_baseline']) for r in j['records'])}
assert max(audit.values())<1e-4
rows=[]
for f in [0,1]:
 rr=[r for r in j['records'] if r['format']==f and r['delta']==.5];items=sorted({r['item'] for r in rr});gaps={}
 for arm in ['none','source','query','both']:
  x=[r for r in rr if r['arm']==arm];gaps[arm]=[]
  for i in items:
   cells={c:np.mean([r['intended_logprob_response'] for r in x if r['item']==i and r['context_same_key']==c]) for c in [False,True]};gaps[arm].append(cells[False]-cells[True])
  rows.append({'format':f,'arm':arm,'context_gap':boot(gaps[arm]),'change_from_none':boot(np.array(gaps[arm])-gaps['none']),'different_context_mean':float(np.mean([r['intended_logprob_response'] for r in x if not r['context_same_key']])),'same_context_mean':float(np.mean([r['intended_logprob_response'] for r in x if r['context_same_key']])),'accuracy':float(np.mean([r['perturbed_full_correct'] for r in x]))})
(A/f'{S}_comparisons.json').write_text(json.dumps({'summary':rows,'audits':audit,'limitations':'Item bootstrap after averaging query and local state. Descriptive reused panel, no multiplicity correction. A path-response transfer tests that path in recipient background, not complete mechanism or unique causal shares.'},indent=2)+'\n');print(audit)
for r in rows:print(r)

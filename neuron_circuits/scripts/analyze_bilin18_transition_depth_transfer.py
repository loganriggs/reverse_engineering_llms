from pathlib import Path
import json,numpy as np
A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition');S='bilin18_transition_depth_transfer';j=json.loads((A/f'{S}.json').read_text());rng=np.random.default_rng(804);summary=[]
def boot(x):
 x=np.asarray(x);b=x[rng.integers(len(x),size=(4096,len(x)))].mean(1);return {'mean':float(x.mean()),'CI':np.quantile(b,[.025,.975]).tolist()}
for f in [0,1]:
 rr=[r for r in j['records'] if r['format']==f and r['delta']==.5];items=sorted({r['item'] for r in rr});baseline=None
 for cut,site in [(-1,'none')]+[(l,s) for l in [4,5,6,7] for s in ['query','all']]:
  x=[r for r in rr if r['cut']==cut and r['site']==site];gaps=[]
  for i in items:
   a=[np.mean([r['response'] for r in x if r['item']==i and r['context_same_key']==c]) for c in [False,True]];gaps.append(a[0]-a[1])
  if baseline is None:baseline=np.array(gaps)
  summary.append({'format':f,'cut':cut,'site':site,'gap':boot(gaps),'gap_change':boot(np.array(gaps)-baseline),'different_mean':float(np.mean([r['response'] for r in x if not r['context_same_key']])),'same_mean':float(np.mean([r['response'] for r in x if r['context_same_key']])),'accuracy':float(np.mean([r['correct'] for r in x]))})
(A/f'{S}_comparisons.json').write_text(json.dumps({'summary':summary,'limitations':'Item bootstrap over query/state averaged contrasts; descriptive and no multiplicity correction. Response swaps retain recipient baselines, input embeddings, and persistent value bus; all-token is not a whole-state swap.'},indent=2)+'\n')
print(j['audits'])
for r in summary:print(r)

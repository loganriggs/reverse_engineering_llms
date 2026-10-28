"""M30 post-result robustness accounting, never a replacement for full-panel scores."""
from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rp=P/'wrapper_crossover_results.json';r=json.loads(rp.read_text());assert json.loads((P/'wrapper_crossover_audit.json').read_text())['result_sha256']==sha(rp);records=r['records'];rows=[]
for kind in ['scalar','block']:
 for seed in [71,72]:
  base=[x for x in records if (x['kind'],x['seed'])==(kind,seed)];initial={(x['target_index'],x['context']):x for x in base if x['arm']=='initial'};ids={i for i,c in initial};families={i:initial[i,'reference']['family'] for i in ids};competent={field:{i for i in ids if all(initial[i,c][field] for c in ['reference','passage','sample'])} for field in ['native_binary','native_top1']}
  for mult in [1.,.1]:
   js={ (x['target_index'],x['context']):x for x in base if x['arm']==f'aggregate_gradient:{mult}:live'};cs={ (x['target_index'],x['context']):x for x in base if x['arm']==f'aggregate_KL:{mult}:live'}
   def ratio(context,keep):
    denominator=sum(cs[i,context]['R'] for i in keep);return sum(js[i,context]['R'] for i in keep)/denominator if denominator else None
   panels={'all':ids,**competent};scores={name:dict(endpoints=len(keep),ratios={c:ratio(c,keep) for c in ['reference','passage','sample']}) for name,keep in panels.items()};loo={f:{c:ratio(c,{i for i in ids if families[i]!=f}) for c in ['reference','passage','sample']} for f in sorted(set(families.values()))};row=dict(kind=kind,seed=seed,multiplier=mult,panels=scores,leave_one_family_out=loo,leave_one_family_out_wins={c:sum(v[c]<1 for v in loo.values() if v[c] is not None) for c in ['reference','passage','sample']});rows.append(row);print(kind,seed,mult,'PANELS',scores,'LOO wins',row['leave_one_family_out_wins'])
out=dict(result_sha256=sha(rp),rows=rows,scope='Post-result descriptive robustness: paired removal of each family, and endpoint subsets with native binary/top1 success in all three wrappers. All-panel result remains primary. These are correlated subsets, not independent replications, significance tests, or selection of a model/task panel.');(P/'wrapper_concentration.json').write_text(json.dumps(out,indent=2)+'\n')

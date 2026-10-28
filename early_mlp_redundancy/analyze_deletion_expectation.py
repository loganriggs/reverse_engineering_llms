"""Post-result analytic expectation check; no fitted parameters or new model calls."""
import itertools,json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent/'validation'
def expectation(total_energy,channel_energy,n,m):
 p=m/n;p2=m*(m-1)/(n*(n-1));return p2*total_energy+(p-p2)*channel_energy
# Independent exhaustive enumeration checks the fixed-size subset formula.
c=np.random.default_rng(6).normal(size=(6,4))
for m in (1,2,3,5,6):
 actual=np.mean([np.square(c[list(indices)].sum(0)).sum() for indices in itertools.combinations(range(6),m)])
 assert abs(actual-expectation(np.square(c.sum(0)).sum(),np.square(c).sum(),6,m))<1e-12
r=json.loads((P/'matched_layers.json').read_text());g={(x['id'],x['endpoint'],x['layer']):x for x in r['geometry']};full={(x['id'],x['endpoint'],x['layer']):x['local_relative_squared_error'] for x in r['rows'] if x['kind']=='all_products'};out=[]
for layer in (0,1,2):
 for frac in (.25,.5,.875):
  rows=[x for x in r['rows'] if x['layer']==layer and x['removed_fraction']==frac];pred=[]
  for x in rows:
   key=(x['id'],x['endpoint'],layer);gg=g[key]
   pred.append(expectation(1,gg['sum_channel_energy']/gg['sum_net_energy'],4608,int(4608*frac))*full[key])
  out.append(dict(layer=layer,removed_fraction=frac,observed_mean_local_error=sum(x['local_relative_squared_error'] for x in rows)/len(rows),analytic_uniform_subset_expected_error=sum(pred)/len(pred)))
result=dict(protocol='Post-result exact expectation diagnostic, not fitted or independent confirmation. Uses per-endpoint energies and all-product relative error for bias-inclusive denominator. Observations use one partition, not an expectation over all masks.',formula_exhaustive_check=True,rows=out)
(P/'matched_layers_deletion_expectation.json').write_text(json.dumps(result,indent=2)+'\n')
print('Exact subset formula checked by exhaustive enumeration; wrote expectation comparison.')

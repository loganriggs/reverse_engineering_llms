import json,numpy as np
from pathlib import Path
A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition');S='bilin18_featurizer_downstream_confirmation';j=json.loads((A/f'{S}.json').read_text());old=json.loads((A/'bilin18_frozen_featurizer_confirmation.json').read_text());rng=np.random.default_rng(838);summary=[];contrasts=[]
for l in [1,2]:
 for p in [0,8]:
  rr=[r for r in j['records'] if r['layer']==l and r['probe']==p];z=np.array([np.mean([r['KL'] for r in rr if r['document']==i and r['model']=='zero']) for i in range(32)]);vals={}
  for kind in ['scalar48','scalar64','block8','shared_fixed','shared_learned','routed_weighted64']:
   match=lambda n:n==kind if kind=='routed_weighted64' else n.startswith(kind+'_')
   v=np.array([np.mean([r['KL'] for r in rr if r['document']==i and match(r['model'])]) for i in range(32)]);vals[kind]=v;loc=[r for r in old['records'] if r['layer']==l and r['probe']==p and r['document']<32 and match(r['model'])];summary.append({'layer':l,'probe':p,'kind':kind,'mean_KL_over_zero':float(v.mean()/z.mean()),'documents_better_than_zero':int((v<z).sum()),'local_relative_RMSE_same_subset':float(np.sqrt(sum(r['squared_error'] for r in loc)/sum(r['truth_squared_norm'] for r in loc)))})
  draw=rng.integers(32,size=(4096,32))
  for a,b in [('scalar48','scalar64'),('scalar48','block8'),('scalar48','shared_fixed'),('scalar64','routed_weighted64')]:
   diff=vals[a]-vals[b];boot=diff[draw].mean(1);contrasts.append({'layer':l,'probe':p,'contrast':a+' minus '+b+' KL (seed mean)','mean':float(diff.mean()),'document_bootstrap95':np.quantile(boot,[.025,.975]).tolist()})
print(json.dumps({'summary':summary,'contrasts':contrasts},indent=2));(A/f'{S}_analysis.json').write_text(json.dumps({'summary':summary,'contrasts':contrasts,'limitations':'Paired32-document bootstrap, averaging signs and seeds first. Seeds not independent replication; no multiplicity correction.'},indent=2)+'\n')

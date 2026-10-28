import json,numpy as np,torch
from pathlib import Path
A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition');S='bilin18_context_operator_bases';j=json.loads((A/f'{S}.json').read_text());old=json.loads((A/'bilin18_operator_state_prediction.json').read_text());d=torch.load(A/'bilin18_operator_state_prediction_captures.pt',weights_only=False);ids=d['validation_documents'];counts=dict(zip(*[v.tolist() for v in torch.unique(d['train_token_ids'],return_counts=True)]));freq={doc:counts.get(int(tok),0) for doc,tok in zip(ids,d['validation_token_ids'])};rng=np.random.default_rng(833);contrasts=[];strata=[]
for l in [1,2]:
 rows=[r for r in j['records']+old['records'] if r['layer']==l];models={r['model'] for r in rows};lookup={m:{doc:[0.,0.] for doc in ids} for m in models}
 for r in rows:
  a=lookup[r['model']][r['document']];a[0]+=r['squared_error'];a[1]+=r['truth_squared_norm']
 arr={m:np.array([lookup[m][doc] for doc in ids]) for m in models};draw=rng.integers(512,size=(2048,512))
 for rank in [4,16,64]:
  for a,b in [(f'response_weighted{rank}',f'residual_pca{rank}'),(f'residual_pca{rank}',f'pca{rank}_radius')]:
   x=arr[a][draw].sum(1);y=arr[b][draw].sum(1);diff=np.sqrt(x[:,0]/x[:,1])-np.sqrt(y[:,0]/y[:,1]);contrasts.append({'layer':l,'comparison':a+' minus '+b,'document_bootstrap95':np.quantile(diff,[.025,.975]).tolist()})
 for label,sel in [('unseen',[i for i,doc in enumerate(ids) if freq[doc]==0]),('seen',[i for i,doc in enumerate(ids) if freq[doc]>0])]:
  for name in ['token_shape_radius','pca64_radius','residual_pca64','response_weighted64']:
   x=arr[name][sel].sum(0);strata.append({'layer':l,'stratum':label,'documents':len(sel),'model':name,'relative_RMSE':float(np.sqrt(x[0]/x[1]))})
(A/f'{S}_analysis.json').write_text(json.dumps({'contrasts':contrasts,'strata':strata,'note':'Development-validation document bootstrap, fixed probe set; seen/unseen breakdown added posthoc.'},indent=2)+'\n');print(json.dumps({'contrasts':contrasts,'strata':strata},indent=2))

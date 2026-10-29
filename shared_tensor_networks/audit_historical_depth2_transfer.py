import json
from pathlib import Path
import numpy as np
from larger_hidden_transfer import values
root=Path('results/larger_hidden_transfer');src=Path('results/scaling_grid');historical=json.loads((src/'results.json').read_text())['rows'];records=[]
for file in sorted(root.glob('g*_*.npz')):
 model=next((m for m in ['shared_budget','oracle_source','separate','shared'] if file.stem.endswith('_'+m)),None)
 if model=='oracle_source':continue
 key=file.stem[:-(len(model)+1)];z=np.load(file);old=np.load(src/key/f'{model}_evaluation.npz');data=np.load(src/key/'training.npz')
 X=values(z['factors'],data['adaptation_inputs'])@z['source_map']/old['feature_norm']
 for j,n in enumerate([32,128]):
  U,s,Vh=np.linalg.svd(X[:n],full_matrices=False);fit=Vh.T@((s/(s*s+n*.001))[:,None]*(U.T@data['adaptation_targets'][:n]))
  np.testing.assert_allclose(X@fit,X@old['B'][j],rtol=1e-7,atol=1e-8)
 previous=next(r for r in historical if r['key']==key and r['model']==model)
 mc=np.array(previous['transfer_relative_mse'])[:2]
 records.append(dict(key=key,model=model,exact=z['historical_relative_mse'].tolist(),historical_monte_carlo=mc.tolist(),max_absolute_mc_difference=float(np.max(np.abs(mc-z['historical_relative_mse'])))))
(root/'historical_audit.json').write_text(json.dumps(dict(status='passed',models=len(records),records=records),indent=2)+'\n')
print('Replayed36 historical models /72 readouts; maximum MC versus exact difference',max(r['max_absolute_mc_difference'] for r in records))

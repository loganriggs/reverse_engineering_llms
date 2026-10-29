"""Exact finite-change identity for normalization-induced ridge penalties."""
import json
from pathlib import Path
import numpy as np
import torch
from depth3_hidden_transfer import numpy_features
root=Path('results/depth3_measurement');src=Path('results/scaling_grid');rows=json.loads((root/'results.json').read_text());assert len(rows)==12
records=[]
for row in rows:
 key,name=row['key'],row['model'];folder=src/key;state={k:v.numpy() for k,v in torch.load(folder/('teacher.pt' if name=='oracle_source' else name+'.pt'),weights_only=True).items()};data=np.load(folder/'training.npz');z=np.load(root/f'{key}_{name}.npz');f,p=numpy_features(state,data['adaptation_inputs']);F=np.concatenate([p[:,:12],f],axis=1)
 for typ,sl in enumerate([slice(0,12),slice(12,None)]):
  X=F[:,sl];d0=z['norms'][0,sl];c0=z[f'coef_{typ}']/d0[:,None]
  for cal in [1,2]:
   d1=z['norms'][cal,sl];c1=z[f'coef_{2*cal+typ}']/d1[:,None];delta=c1-c0
   # H1 delta +lambda(D1²-D0²)c0=0; no inverse or p-by-p Gram needed.
   first=X.T@(X@delta)/128+.001*d1[:,None]**2*delta
   rhs=.001*(d1*d1-d0*d0)[:,None]*c0
   error=np.linalg.norm(first+rhs)/max(1.,np.linalg.norm(first),np.linalg.norm(rhs));assert error<1e-7,error
   normalized_obj=np.mean((X@(z[f'coef_{2*cal+typ}']/d1[:,None])-data['adaptation_targets'])**2)*4+.001*np.sum(z[f'coef_{2*cal+typ}']**2)
   raw_obj=np.sum((X@c1-data['adaptation_targets'])**2)/128+.001*np.sum((d1[:,None]*c1)**2)
   np.testing.assert_allclose(normalized_obj,raw_obj,rtol=1e-12,atol=1e-12)
   records.append(dict(key=key,model=name,representation=typ,calibration=cal,resolvent_residual=float(error)))
(root/'penalty_checks.json').write_text(json.dumps(records,indent=2)+'\n');print('48 finite-change penalty identities passed.')

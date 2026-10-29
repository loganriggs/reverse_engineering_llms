import json
from pathlib import Path
import numpy as np
from multitask import monomials,product_map
from experiment import gram
root=Path('results/variable_projection/development');rows=json.loads((root/'results.json').read_text())
keys=monomials(6);G=gram(keys);results=[]
def coeff(v):return product_map(v[:18].reshape(3,6),v[18:36].reshape(3,6),keys)@v[36:].reshape(6,6).T
for row in rows:
 z=np.load(root/f"{row['seed']}_{row['method']}.npz");v=z['final'];C=coeff(v);T=coeff(z['teacher'])
 design=np.prod(z['x'][:,None,:]**np.array(keys)[None,:,:],axis=2)
 pred=design@C;loss=np.mean((pred-z['y'])**2)
 np.testing.assert_allclose(loss,row['train_mse'],rtol=1e-8,atol=1e-10)
 delta=C-T;population=np.sum(delta*(G@delta))/6
 a,b=v[:18].reshape(3,6),v[18:36].reshape(3,6)
 hidden=(z['x']@a.T)*(z['x']@b.T)
 phi=np.stack([hidden[:,i]*hidden[:,j] for i in range(3) for j in range(i,3)],axis=1)
 Q,R=np.linalg.qr(phi,mode='reduced');fitted=Q@(Q.T@z['y'])
 gap=np.linalg.norm(pred-fitted)/max(1.,np.linalg.norm(fitted))
 if row['method']=='reduced':assert gap<1e-8,gap
 results.append(dict(seed=row['seed'],method=row['method'],population_mse=float(population),
                     readout_QR_prediction_gap=gap,design_condition=float(np.linalg.cond(phi)),
                     max_hidden_parameter=float(np.max(np.abs(v[:36])))))
(root/'audit.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))

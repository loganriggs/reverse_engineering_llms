import hashlib,itertools,json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from multitask import monomials,product_map
from experiment import gram
from direct_bilinear_readout import quadratic_coefficients
root=Path('results/crossed_readout');src=Path('results/normalized_readout/panel')
rows=json.loads((root/'results.json').read_text());assert len(rows)==60
keys=monomials(8);G=gram(keys);qkeys=[tuple(int(k==i)+int(k==j) for k in range(8)) for i,j in itertools.combinations_with_replacement(range(8),2)];Gq=gram(qkeys);records=[]
for row in rows:
 file=root/f"{row['seed']}_{row['init']}_{row['condition']}_{row['method']}.npz";datafile=src/f"{row['seed']}_data.npz"
 assert hashlib.sha256(file.read_bytes()).hexdigest()==row['checkpoint_sha256']
 assert hashlib.sha256(datafile.read_bytes()).hexdigest()==row['data_sha256']
 z=np.load(file);data=np.load(datafile);raw=z['final'][:64].reshape(8,8);norm=np.linalg.norm(raw,axis=1)
 ab=raw/norm[:,None] if row['normalized'] else raw
 a,b=ab[:4],ab[4:];U=z['readout'];x=data['x']
 def features(x):
  hi=(x@a.T)*(x@b.T)
  return np.stack([hi[:,i]*hi[:,j] for i,j in itertools.combinations_with_replacement(range(4),2)],axis=1)
 phi=features(x);Z=np.concatenate([phi,np.sqrt(row['ridge'])*np.eye(10)]);Y=np.concatenate([data['y'],np.zeros((10,10))])
 Q,R=np.linalg.qr(Z,mode='reduced');best=np.linalg.solve(R,Q.T@Y)
 optimal_prediction_error=np.linalg.norm(Z@U-Q@(Q.T@Y))/max(1.,np.linalg.norm(Y))
 if row['method']=='reduced':assert optimal_prediction_error<1e-7,optimal_prediction_error
 radial=.5*np.sum((norm-1)**2) if row['normalized'] else 0.
 obj=.5*np.sum((Z@U-Y)**2)+radial;np.testing.assert_allclose(obj,row['objective'],rtol=1e-8,atol=1e-8)
 F=product_map(a,b,keys);np.testing.assert_allclose(F,z['features'],rtol=1e-12,atol=1e-10)
 poly=np.prod(x[:,None,:]**np.array(keys)[None,:,:],axis=2)@F@U
 forward_error=np.linalg.norm(poly-phi@U)/max(1.,np.linalg.norm(poly));assert forward_error<1e-7
 e=F@U-data['truth'][:,:10];source=np.sum(e*(G@e))/10;np.testing.assert_allclose(source,row['source_mse'],rtol=1e-8,atol=1e-10)
 fa=features(data['xa']);fit,_,rank,_=np.linalg.lstsq(fa,data['ya'],rcond=None)
 np.testing.assert_allclose(fa@fit,fa@z['transfer'],rtol=1e-6,atol=1e-8)
 e=F@fit-data['truth'][:,10:];transfer=np.sum(e*(G@e))/4
 np.testing.assert_allclose(transfer,row['transfer_mse'],rtol=1e-6,atol=1e-8)
 c=quadratic_coefficients(a,b);truth=quadratic_coefficients(data['teacher_A'],data['teacher_B'])
 c/=np.sqrt(np.sum(c*(Gq@c),axis=0));truth/=np.sqrt(np.sum(truth*(Gq@truth),axis=0))
 cos=np.abs(c.T@Gq@truth);i,j=linear_sum_assignment(-cos)
 records.append(dict(seed=row['seed'],init=row['init'],condition=row['condition'],method=row['method'],
  optimal_prediction_error=optimal_prediction_error,forward_error=forward_error,adaptation_rank=int(rank),
  min_component_cosine=float(min(cos[i,j])),all_components_above_099=bool(min(cos[i,j])>.99)))
(root/'audit.json').write_text(json.dumps(records,indent=2)+'\n');print('Audited60 source/transfer fits, all adaptation ranks:',sorted(set(r['adaptation_rank'] for r in records)))

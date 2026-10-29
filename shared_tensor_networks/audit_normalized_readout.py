import hashlib,json,itertools
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from multitask import monomials,product_map
from experiment import gram
from direct_bilinear_readout import quadratic_coefficients
root=Path('results/normalized_readout/panel');rows=json.loads((root/'results.json').read_text());assert len(rows)==20
keys=monomials(8);G=gram(keys)
qkeys=[tuple(int(k==i)+int(k==j) for k in range(8)) for i,j in itertools.combinations_with_replacement(range(8),2)];Gq=gram(qkeys)
records=[]
for row in rows:
 data=np.load(root/f"{row['seed']}_data.npz");file=root/f"{row['seed']}_{row['init']}_{row['method']}.npz";z=np.load(file)
 raw=z['final'][:64].reshape(8,8);norm=np.linalg.norm(raw,axis=1);unit=raw/norm[:,None];U=z['readout']
 x=data['x'];a,b=unit[:4],unit[4:];hidden=(x@a.T)*(x@b.T)
 phi=np.stack([hidden[:,i]*hidden[:,j] for i,j in itertools.combinations_with_replacement(range(4),2)],axis=1)
 Z=np.concatenate([phi,np.sqrt(.001)*np.eye(10)]);Y=np.concatenate([data['y'],np.zeros((10,10))])
 Q,R=np.linalg.qr(Z,mode='reduced');best=np.linalg.solve(R,Q.T@Y)
 normal=np.linalg.norm(Z.T@(Z@U-Y))/max(1.,np.linalg.norm(Z)*np.linalg.norm(Z@U-Y))
 if row['method']=='reduced':np.testing.assert_allclose(Z@U,Q@(Q.T@Y),rtol=1e-8,atol=1e-9)
 obj=.5*np.sum((Z@U-Y)**2)+.5*np.sum((norm-1)**2)
 np.testing.assert_allclose(obj,row['objective'],rtol=1e-9,atol=1e-9)
 F=product_map(a,b,keys);np.testing.assert_allclose(F,z['features'],atol=1e-12)
 px=np.prod(x[:,None,:]**np.array(keys)[None,:,:],axis=2)
 np.testing.assert_allclose(px@F,phi,rtol=1e-8,atol=1e-10)
 err=F@U-data['truth'][:,:10];src=np.sum(err*(G@err))/10
 np.testing.assert_allclose(src,row['source_mse'],rtol=1e-9,atol=1e-12)
 xa=data['xa'];ha=(xa@a.T)*(xa@b.T);fa=np.stack([ha[:,i]*ha[:,j] for i,j in itertools.combinations_with_replacement(range(4),2)],axis=1)
 transfer=np.linalg.lstsq(fa,data['ya'],rcond=None)[0]
 np.testing.assert_allclose(fa@transfer,fa@z['transfer'],rtol=1e-8,atol=1e-10)
 e=F@transfer-data['truth'][:,10:];te=np.sum(e*(G@e))/4
 np.testing.assert_allclose(te,row['transfer_mse'],rtol=1e-8,atol=1e-10)
 c=quadratic_coefficients(a,b);truth=quadratic_coefficients(data['teacher_A'],data['teacher_B'])
 cosine=np.abs(c.T@Gq@truth)/np.sqrt(np.outer(np.sum(c*(Gq@c),axis=0),np.sum(truth*(Gq@truth),axis=0)))
 i,j=linear_sum_assignment(-cosine);matched=cosine[i,j]
 records.append(dict(seed=row['seed'],init=row['init'],method=row['method'],normal_relative=normal,
   min_component_cosine=float(min(matched)),all_components_above_099=bool(min(matched)>.99),
   checkpoint_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),profiled_objective=float(.5*np.sum((Z@best-Y)**2)+.5*np.sum((norm-1)**2))))
# Both starts use identical actual predictions and optimal readouts.
for seed in range(232101,232106):
 for init in [233001,233002]:
  joint=np.load(root/f'{seed}_{init}_joint.npz');reduced=np.load(root/f'{seed}_{init}_reduced.npz')
  np.testing.assert_array_equal(joint['initial'][:64],reduced['initial'])
  data=np.load(root/f'{seed}_data.npz');u=joint['initial'][:64].reshape(8,8);u=u/np.linalg.norm(u,axis=1)[:,None]
  hi=(data['x']@u[:4].T)*(data['x']@u[4:].T)
  pi=np.stack([hi[:,i]*hi[:,j] for i,j in itertools.combinations_with_replacement(range(4),2)],axis=1)
  zi=np.concatenate([pi,np.sqrt(.001)*np.eye(10)]);yi=np.concatenate([data['y'],np.zeros((10,10))])
  qi,ri=np.linalg.qr(zi,mode='reduced');ui=np.linalg.solve(ri,qi.T@yi)
  np.testing.assert_allclose(zi@joint['initial'][64:].reshape(10,10).T,zi@ui,atol=1e-10,rtol=1e-10)
# Verify the exact profiling identity at final and arbitrary perturbed readouts.
for row in rows:
 z=np.load(root/f"{row['seed']}_{row['init']}_{row['method']}.npz");data=np.load(root/f"{row['seed']}_data.npz")
 u=z['final'][:64].reshape(8,8);norm=np.linalg.norm(u,axis=1);assert max(abs(norm-1))<1e-8
 u=u/norm[:,None];hidden=(data['x']@u[:4].T)*(data['x']@u[4:].T)
 phi=np.stack([hidden[:,i]*hidden[:,j] for i,j in itertools.combinations_with_replacement(range(4),2)],axis=1)
 Z=np.concatenate([phi,np.sqrt(.001)*np.eye(10)]);Y=np.concatenate([data['y'],np.zeros((10,10))]);Q,R=np.linalg.qr(Z,mode='reduced');best=np.linalg.solve(R,Q.T@Y)
 U=z['readout']+np.random.default_rng(row['seed']).normal(size=(10,10))*.1
 lhs=np.sum((Z@U-Y)**2);rhs=np.sum((Z@best-Y)**2)+np.sum((Z@(U-best))**2)
 np.testing.assert_allclose(lhs,rhs,rtol=1e-11,atol=1e-9)
(root/'audit.json').write_text(json.dumps(records,indent=2)+'\n')
for method in ['joint','reduced']:
 rr=[r for r in records if r['method']==method]
 print(method,'component recovery',sum(r['all_components_above_099'] for r in rr),'/10', 'max normal residual',max(r['normal_relative'] for r in rr))
print('Audited 20 fits and transfers.')

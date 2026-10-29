"""Replay empirical activation norms/subspaces without population-metric extraction."""
import itertools,json
from pathlib import Path
import numpy as np
from experiment import gram
from multitask import monomials,product_map
from run_svhn_standard import digest,write
root=Path(__file__).parent;out=root/'results/empirical_extraction';assert json.loads((out/'completed.json').read_text())['representations']==300
protocol=json.loads((out/'protocol.json').read_text());assert digest(out/'source.py')==protocol['source_sha256']
keys=monomials(8);G=gram(keys);L=np.linalg.cholesky(G).T;qpairs=list(itertools.combinations_with_replacement(range(8),2));qkeys=[tuple(pair.count(i) for i in range(8)) for pair in qpairs]
def design(x,exps):return np.prod(x[:,None,:]**np.array(exps)[None,:,:],axis=2)
cache={};states_cache={};records=[]
for seed in range(225001,225006):
 data=np.load(root/f'results/finite_readout_h4/{seed}_data.npz');fresh=np.load(out/f'{seed}_unlabeled.npz')['inputs'];np.testing.assert_array_equal(fresh,np.random.default_rng(seed+7000).normal(size=(4096,8)))
 cache[seed]=(data,{name:(x,design(x,keys)) for name,x in [('source512',data['inputs'][:512]),('fresh512',fresh[:512]),('fresh4096',fresh)]})
for path in sorted(out.glob('225*_*.json')):
 r=json.loads(path.read_text());d=np.load(path.with_suffix('.npz'));assert digest(path.with_suffix('.npz'))==r['checkpoint_sha256'];source=root/r['input_file'];assert digest(source)==r['input_sha256'];assert digest(root/r['data_file'])==r['data_sha256'];data,pools=cache[r['teacher_seed']];x,pk=pools[r['pool']]
 if r['model_name'] not in states_cache:
  src=np.load(source);parent=json.loads(source.with_suffix('.json').read_text());states=[np.load(root/parent['input_file'])['final']] if r['method']=='shared' else [src[f'final_{i}'] for i in range(10)];states_cache[r['model_name']]=states
 rawvalues=[];rawcoeff=[]
 for state in states_cache[r['model_name']]:
  A=state[:32].reshape(4,8);B=state[32:64].reshape(4,8);z=(x@A.T)*(x@B.T)
  if r['extraction']=='quadratic_consensus':
   rawvalues.append(z);rawcoeff.append(np.stack([A[:,i]*B[:,j]+(A[:,j]*B[:,i] if i!=j else 0) for i,j in qpairs]))
  else:
   rawvalues.append(np.stack([z[:,i]*z[:,j] for i in range(4) for j in range(i,4)],1));rawcoeff.append(product_map(A,B,keys))
 rawvalues=np.concatenate(rawvalues,1);rawcoeff=np.concatenate(rawcoeff,1);norm=np.sqrt(np.mean(rawvalues**2,axis=0));np.testing.assert_allclose(norm,d['raw_column_norms'],atol=1e-9,rtol=1e-9)
 rank=4 if r['extraction']=='quadratic_consensus' else 10;assert r['rank']==rank
 Z=rawvalues/norm/np.sqrt(len(x));eigen=np.linalg.eigvalsh(Z.T@Z)[-rank:][::-1];s=d['singular_values'][:rank];V=d['basis_weights']*s
 np.testing.assert_allclose(s*s,eigen,atol=1e-8,rtol=1e-8);np.testing.assert_allclose(V.T@V,np.eye(rank),atol=1e-9);np.testing.assert_allclose((Z.T@Z)@V,V*eigen,atol=1e-8,rtol=1e-8)
 np.testing.assert_allclose(rawcoeff/norm@d['basis_weights'],d['basis'],atol=1e-9,rtol=1e-9)
 selected=rawvalues/norm@d['basis_weights']
 if rank==4:selected=np.stack([selected[:,i]*selected[:,j] for i in range(4) for j in range(i,4)],1)
 output_norm=np.sqrt(np.mean(selected**2,axis=0));np.testing.assert_allclose(output_norm,d['output_norms'],atol=1e-9,rtol=1e-9)
 np.testing.assert_allclose(pk@d['features'],selected/output_norm,atol=1e-8,rtol=1e-8)
 pa=design(data['adaptation_inputs'],keys);truth=data['truth'][:,10:];F=d['features']
 for item in r['metrics']:
  count=item['labels'];read=d['readout_'+str(count)];X,Y=(L@F,L@truth) if count=='population' else (pa[:count]@F,pa[:count]@truth+.05*data['adaptation_noise'][:count])
  assert np.linalg.matrix_rank(X)==10
  assert np.linalg.norm(X.T@(X@read-Y))/max(1,np.linalg.norm(X.T@Y))<1e-7
  diff=F@read-truth;np.testing.assert_allclose(np.sum(diff*(G@diff),axis=0),item['mse'],atol=1e-8,rtol=1e-8)
 records.append(dict(name=r['name'],result_sha256=digest(path),checkpoint_sha256=digest(path.with_suffix('.npz'))))
assert len(records)==300
write(out/'audit.json',dict(status='passed',representations=300,records=records,source_sha256=digest(Path(__file__))))
print('Audited300 empirical extractions and900 projection/adaptation fits.')

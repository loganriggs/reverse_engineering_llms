"""Verify fixed-rank extraction and transfer independently of the producer."""
import itertools,json
from pathlib import Path
import numpy as np
from experiment import gram
from multitask import monomials
from run_svhn_standard import digest,write
root=Path(__file__).parent;out=root/'results/hidden_extraction';assert json.loads((out/'completed.json').read_text())['models']==50
protocol=json.loads((out/'protocol.json').read_text());assert digest(out/'source.py')==protocol['source_sha256']
keys=monomials(8);G=gram(keys);L=np.linalg.cholesky(G).T
qpairs=list(itertools.combinations_with_replacement(range(8),2));qkeys=[tuple(pair.count(i) for i in range(8)) for pair in qpairs];G2=gram(qkeys);L2=np.linalg.cholesky(G2).T
probe=np.random.default_rng(228001).normal(size=(400,8));phi=np.prod(probe[:,None,:]**np.array(keys)[None,:,:],axis=2);qphi=np.prod(probe[:,None,:]**np.array(qkeys)[None,:,:],axis=2)
records=[]
for path in sorted(out.glob('225*.json')):
 r=json.loads(path.read_text());d=np.load(path.with_suffix('.npz'));assert digest(path.with_suffix('.npz'))==r['checkpoint_sha256'];source=root/r['input_file'];assert digest(source)==r['input_sha256'];src=np.load(source);parent=json.loads(source.with_suffix('.json').read_text())
 audit=json.loads((source.parent/'audit.json').read_text());assert audit['status']=='passed';ref=next(a for a in audit['records'] if a['name']==r['name']);assert ref['checkpoint_sha256']==digest(source)
 assert digest(root/r['data_file'])==r['data_sha256'];data=np.load(root/r['data_file']);truth=data['truth'];pa=np.prod(data['adaptation_inputs'][:,None,:]**np.array(keys)[None,:,:],axis=2)
 states=[np.load(root/parent['input_file'])['final']] if r['method']=='shared' else [src[f'final_{i}'] for i in range(10)]
 directq=[];directproducts=[]
 for state in states:
  z=(probe@state[:32].reshape(4,8).T)*(probe@state[32:64].reshape(4,8).T);directq.append(z);directproducts.append(np.stack([z[:,i]*z[:,j] for i in range(4) for j in range(i,4)],1))
 for values,features,metric in [(np.concatenate(directq,1),d['normalized_quadratics'],G2),(np.concatenate(directproducts,1),d['features_hidden_union'],G)]:
  evals=(qphi if metric is G2 else phi)@features;ratios=np.sum(values*evals,axis=0)/np.sum(evals**2,axis=0)
  assert (ratios>0).all();np.testing.assert_allclose(evals*ratios,values,atol=1e-8,rtol=1e-8);np.testing.assert_allclose(np.sum(features*(metric@features),axis=0),1,atol=1e-10)
 np.testing.assert_array_equal(d['features_outputs'],src['features'])
 Q=d['normalized_quadratics'];cq=d['compressed_quadratics'];wq=L2@Q;basis=L2@cq
 np.testing.assert_allclose(basis.T@basis,np.eye(4),atol=1e-10)
 cov=wq@wq.T;expected=np.linalg.eigvalsh(cov)[-4:][::-1]
 np.testing.assert_allclose(cov@basis,basis*expected,atol=1e-8,rtol=1e-8)
 vals=qphi@cq;products=np.stack([vals[:,i]*vals[:,j] for i in range(4) for j in range(i,4)],1);observed=phi@d['features_quadratic_consensus'];ratio=np.sum(products*observed,0)/np.sum(observed**2,0)
 assert (ratio>0).all();np.testing.assert_allclose(observed*ratio,products,atol=1e-9,rtol=1e-9)
 union=L@d['features_hidden_union'];pc=L@d['features_quartic_pca'];np.testing.assert_allclose(pc.T@pc,np.eye(10),atol=1e-9)
 cov=union@union.T;top=np.linalg.eigvalsh(cov)[-10:][::-1];np.testing.assert_allclose(cov@pc,pc*top,atol=1e-8,rtol=1e-8)
 tq=np.stack([data['teacher_A'][:,i]*data['teacher_B'][:,j]+(data['teacher_A'][:,j]*data['teacher_B'][:,i] if i!=j else 0) for i,j in qpairs]);qt=np.linalg.qr(L2@tq)[0]
 np.testing.assert_allclose(np.linalg.svd(qt.T@basis,compute_uv=False),r['consensus_principal_cosines'],atol=1e-9,rtol=1e-9)
 for item in r['metrics']:
  F=d['features_'+item['extraction']];count=item['labels'];read=d[f"readout_{item['extraction']}_{count}"]
  X,Y=(L@F,L@truth[:,10:]) if count=='population' else (pa[:count]@F,pa[:count]@truth[:,10:]+.05*data['adaptation_noise'][:count])
  assert np.linalg.norm(X.T@(X@read-Y))/max(1,np.linalg.norm(X.T@Y))<1e-7
  u,s,vh=np.linalg.svd(X,full_matrices=False);vr=vh[s>s[0]*1e-12].T;np.testing.assert_allclose(vr@(vr.T@read),read,atol=1e-7,rtol=1e-7)
  delta=F@read-truth[:,10:];np.testing.assert_allclose(np.sum(delta*(G@delta),axis=0),item['mse'],atol=1e-7,rtol=1e-7)
 records.append(dict(name=r['name'],result_sha256=digest(path),checkpoint_sha256=digest(path.with_suffix('.npz'))))
assert len(records)==50
write(out/'audit.json',dict(status='passed',models=50,records=records,source_sha256=digest(Path(__file__))))
print('Audited50 hidden-extraction models and600 adaptation/projection fits.')

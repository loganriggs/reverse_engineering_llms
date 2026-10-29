"""Independent per-head forward/autograd, source moments and output-span transfer."""
import itertools,json
from pathlib import Path
import numpy as np
import torch
from experiment import gram
from multitask import monomials,product_map
from direct_bilinear_readout import predict_jacobian
from run_svhn_standard import digest,write

torch.set_num_threads(1);root=Path(__file__).parent;out=root/'results/released_readouts';assert json.loads((out/'completed.json').read_text())['individual_head_fits']==200
protocol=json.loads((out/'protocol.json').read_text());assert digest(out/'source.py')==protocol['source_sha256']
keys=monomials(8);G=gram(keys);L=np.linalg.cholesky(G).T
baseaudit=json.loads((root/'results/finite_readout_h4/audit.json').read_text());assert baseaudit['status']=='passed'
probe=np.random.default_rng(227001).normal(size=(400,8));phip=np.prod(probe[:,None,:]**np.array(keys)[None,:,:],axis=2)
records=[];worst=0.;heads=0
for path in sorted(out.glob('225*.json')):
 r=json.loads(path.read_text());p=path.with_suffix('.npz');assert digest(p)==r['checkpoint_sha256'];d=np.load(p)
 source=root/r['input_file'];assert digest(source)==r['input_sha256'];assert digest(root/r['data_file'])==r['data_sha256'];src=np.load(source);data=np.load(root/r['data_file'])
 ref=next(a for a in baseaudit['records'] if a['name']==source.stem);assert ref['checkpoint_sha256']==digest(source)
 truth=data['truth'];norms=data['norms'];x=data['inputs'][:512];phi=np.prod(x[:,None,:]**np.array(keys)[None,:,:],axis=2);pa=np.prod(data['adaptation_inputs'][:,None,:]**np.array(keys)[None,:,:],axis=2);y=phi@truth[:,:10]+.05*data['noise'][:512,:10]
 columns=[]
 initial_shared=src['final'] if r['method']=='released' else np.concatenate([data['teacher_A'].ravel(),data['teacher_B'].ravel(),data['teacher_W'][:10].ravel()])
 h=4;n=8*h;q=h*(h+1)//2;pairs=list(itertools.combinations_with_replacement(range(h),2));assert r['parameters']==10*(2*n+q) and len(r['head_records'])==10
 for t,hr in enumerate(r['head_records']):
  expected=np.concatenate([initial_shared[:32].reshape(4,8)[:h].ravel(),initial_shared[32:64].reshape(4,8)[:h].ravel(),initial_shared[64:].reshape(10,10)[t,:q]])
  np.testing.assert_array_equal(d[f'initial_{t}'],expected)
  for stage in ['initial','final']:
   v=d[f'{stage}_{t}'];A=v[:n].reshape(h,8);B=v[n:2*n].reshape(h,8);W=v[2*n:];K=product_map(A,B,keys);co=K@W/norms[t]
   direct=((probe@A.T)*(probe@B.T));products=np.stack([direct[:,i]*direct[:,j] for i,j in pairs],axis=1)
   np.testing.assert_allclose(phip@co,products@W/norms[t],atol=1e-8,rtol=1e-8)
   tx=torch.tensor(probe[:13])
   def forward(v):
    z=(tx@v[:n].reshape(h,8).T)*(tx@v[n:2*n].reshape(h,8).T);pp=torch.stack([z[:,i]*z[:,j] for i,j in pairs],1)
    return pp@v[2*n:]
   auto=torch.func.jacrev(forward)(torch.tensor(v)).numpy();actual=predict_jacobian(v,probe[:13],h)[1][:,0,:]
   error=float(np.linalg.norm(auto-actual)/max(1,np.linalg.norm(auto)));assert error<1e-8;worst=max(worst,error)
  initial_prediction=predict_jacobian(d[f'initial_{t}'],x,h,False)[:,0]/norms[t]
  initial_loss=float(np.mean((initial_prediction-y[:,t])**2))
  np.testing.assert_allclose(initial_loss,hr['initial_train_mse'],atol=1e-9,rtol=1e-9)
  assert hr['train_mse']<=initial_loss+1e-9
  columns.append(co);train=float(np.mean((phi@co-y[:,t])**2));assert abs(train-hr['train_mse'])<1e-8;heads+=1
 C=np.stack(columns,axis=1)
 np.testing.assert_allclose(C,d['coefficients'],atol=1e-10,rtol=1e-10);delta=C-truth[:,:10];np.testing.assert_allclose(np.sum(delta*(G@delta),axis=0),r['source_mse'],atol=1e-8,rtol=1e-8)
 F=C/np.sqrt(np.sum(C*(G@C),axis=0));np.testing.assert_allclose(F,d['features'],atol=1e-10,rtol=1e-10)
 for kind,features in [('learned',F),('oracle',truth[:,:10])]:
  for count in ['population',32,128]:
   key=f'{kind}_{count}';read=d['readout_'+key];X,Y=(L@features,L@truth[:,10:]) if count=='population' else (pa[:count]@features,pa[:count]@truth[:,10:]+.05*data['adaptation_noise'][:count])
   assert np.linalg.norm(X.T@(X@read-Y))/max(1,np.linalg.norm(X.T@Y))<1e-7
   diff=features@read-truth[:,10:];np.testing.assert_allclose(np.sum(diff*(G@diff),axis=0),r['transfer_mse'][key],atol=1e-8,rtol=1e-8)
 records.append(dict(name=r['name'],result_sha256=digest(path),checkpoint_sha256=digest(p)))
assert len(records)==20 and heads==200
write(out/'audit.json',dict(status='passed',groups=20,individual_head_fits=heads,max_jacobian_error=worst,records=records,source_sha256=digest(Path(__file__))))
print('Audited20groups/200headfits',worst)

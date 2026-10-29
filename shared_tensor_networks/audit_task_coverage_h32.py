import hashlib,itertools,json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from quartic_cross_gram import product_factors,cross_gram
from direct_bilinear_readout import quadratic_coefficients
from experiment import gram
from coverage_h32_data import load_data
root=Path('results/task_coverage_h32');out=root/'panel';rows=json.loads((out/'results.json').read_text());assert len(rows)==60
def quadratic_gram(d):
 pairs=list(itertools.combinations_with_replacement(range(d),2));G=np.eye(len(pairs));diagonal=[k for k,(i,j) in enumerate(pairs) if i==j]
 G[np.ix_(diagonal,diagonal)]=np.ones((d,d))+2*np.eye(d)
 return G
for d in [3,4]:
 keys=[tuple(int(k==i)+int(k==j) for k in range(d)) for i,j in itertools.combinations_with_replacement(range(d),2)]
 np.testing.assert_array_equal(quadratic_gram(d),gram(keys))
Gq=quadratic_gram(64);audit=[]
def phi(A,B,x):
 z=(x@A.T)*(x@B.T);return np.stack([z[:,i]*z[:,j] for i,j in itertools.combinations_with_replacement(range(32),2)],axis=1)
for seed in range(251101,251106):
 data=load_data(out,seed);TA,TB,TW=data['teacher_A'],data['teacher_B'],data['teacher_W'];TF=product_factors(TA,TB,np.ones(32));TG=cross_gram(TF,TF);energy=np.einsum('tp,pq,tq->t',TW,TG,TW);np.testing.assert_allclose(energy,1,atol=1e-10)
 np.testing.assert_allclose(phi(TA,TB,data['x'])@TW[:528].T+.05*data['noise'],data['y'],rtol=1e-10,atol=1e-10)
 np.testing.assert_allclose(phi(TA,TB,data['xa'])@TW[528:].T+.05*data['adaptation_noise'],data['ya'],rtol=1e-10,atol=1e-10)
 curves=json.loads((out/f'{seed}_curves.json').read_text());assert [x['step'] for x in curves]==list(range(200,3001,200));assert all(len(x['preupdate_loss'])==12 for x in curves)
 for row in [x for x in rows if x['teacher']==seed]:
  file=out/f"{row['name']}.npz";z=np.load(file);A,B,W=z['A'],z['B'],z['W'];n,t=row['n'],row['tasks'];F=product_factors(A,B,np.ones(32));G=cross_gram(F,F);C=cross_gram(F,TF)
  np.testing.assert_array_equal(W[t:],z['initial_W'][t:].astype(float))
  if row['initialization']=='teacher':
   np.testing.assert_array_equal(z['initial_A'],TA.astype(np.float32));np.testing.assert_array_equal(z['initial_B'],TB.astype(np.float32));np.testing.assert_array_equal(z['initial_W'],TW[:528].astype(np.float32))
  pred=phi(A,B,data['x'][:n])@W[:t].T;train=np.mean((pred-data['y'][:n,:t])**2);np.testing.assert_allclose(train,row['train_mse'],rtol=1e-8,atol=1e-10)
  err=np.einsum('tp,pq,tq->t',W[:t],G,W[:t])-2*np.sum((W[:t]@C)*TW[:t],axis=1)+energy[:t];assert min(err)>-1e-7;np.testing.assert_allclose(err,row['population_mse'],rtol=1e-7,atol=1e-9)
  for k in [512,2048]:
   for typ,a,b,gg,cc in [('learned',A,B,G,C),('oracle',TA,TB,TG,TG)]:
    X=phi(a,b,data['xa'][:k]);coef=z[f'{typ}_{k}'];u,s,vh=np.linalg.svd(X,full_matrices=False);rank=s>s[0]*1e-12;ref=vh[rank].T@((u[:,rank].T@data['ya'][:k])/s[rank,None]);np.testing.assert_allclose(coef,ref,rtol=1e-7,atol=1e-9)
    error=np.sum(coef*(gg@coef),axis=0)-2*np.sum(coef*(cc@TW[528:].T),axis=0)+energy[528:];np.testing.assert_allclose(error,row['transfer'][f'{typ}_{k}'],rtol=1e-7,atol=1e-9)
  q=quadratic_coefficients(A,B);qt=quadratic_coefficients(TA,TB);q/=np.sqrt(np.sum(q*(Gq@q),axis=0));qt/=np.sqrt(np.sum(qt*(Gq@qt),axis=0));cos=abs(q.T@Gq@qt);ii,jj=linear_sum_assignment(-cos)
  np.testing.assert_allclose(cos[ii,jj],row['component_cosines'],rtol=1e-8,atol=1e-10)
  audit.append(dict(name=row['name'],checkpoint_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),data_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [out/f'{seed}_data.npz']+[out/f'{seed}_source_{i}.npz' for i in range(4)]}))
 # Exact initial pairing across counts and all cases is tested from saved float32 state.
 for init in ['252001','252002','teacher']:
  files=[np.load(out/f'{seed}_n{n}_t{t}_{init}.npz') for n in [2048,8192] for t in [132,528]]
  for z in files[1:]:
   for key in ['initial_A','initial_B','initial_W']:np.testing.assert_array_equal(z[key],files[0][key])
(root/'audit.json').write_text(json.dumps(dict(status='passed',fits=60,transfer_fits=240,records=audit),indent=2)+'\n');print('Audited60source fits,240transfer fits,all component metrics/data/initialization pairing.')

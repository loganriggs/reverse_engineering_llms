import hashlib,itertools,json
from pathlib import Path
import numpy as np
import torch
from scipy.linalg import eigh
from deep_multitask import DeepMultiTask
from quartic_cross_gram import cross_gram
from larger_hidden_transfer import sphere_factor,values,target_moments
root=Path('results/larger_hidden_transfer');src=Path('results/scaling_grid');rows=json.loads((root/'results.json').read_text());assert len(rows)==192
records=[];diagnostics=[]
for key,model in sorted(set((r['key'],r['model']) for r in rows)):
 rr=[r for r in rows if r['key']==key and r['model']==model];folder=src/key;data=np.load(folder/'training.npz');saved=np.load(root/f'{key}_{model}.npz');F,T=saved['factors'],saved['source_map'];d=F.shape[-1]
 file=folder/('teacher.pt' if model=='oracle_source' else model+'.pt');state=torch.load(file,weights_only=True)
 assert all(hashlib.sha256(file.read_bytes()).hexdigest()==r['checkpoint_sha256'] for r in rr)
 assert all(hashlib.sha256((folder/'training.npz').read_bytes()).hexdigest()==r['data_sha256'] for r in rr)
 shared=state['A.0'].ndim==2;h=state['A.0'].shape[-2];tasks=state['W'].shape[0]
 network=DeepMultiTask(d,h,tasks,2,shared,0);network.load_state_dict(state)
 x=data['adaptation_inputs'];y=data['adaptation_targets'];X=values(F,x)
 with torch.no_grad():
  direct=network(torch.tensor(x)).numpy()[:,:12]
 np.testing.assert_allclose(X@T,direct,rtol=1e-8,atol=1e-10)
 # Literal hidden product activations independently recover archived feature order.
 with torch.no_grad():hidden=network.hidden(torch.tensor(x)).numpy()
 if shared:literal=np.stack([hidden[:,i]*hidden[:,j] for i,j in itertools.combinations_with_replacement(range(h),2)],axis=1)
 else:literal=np.concatenate([np.stack([hidden[:,t,i]*hidden[:,t,j] for i,j in itertools.combinations_with_replacement(range(h),2)],axis=1) for t in range(12)],axis=1)
 np.testing.assert_allclose(X,literal,rtol=1e-8,atol=1e-10)
 G=cross_gram(F,F)*sphere_factor(d);teacher=torch.load(folder/'teacher.pt',weights_only=True);C,E=target_moments(F,teacher,d)
 np.testing.assert_allclose(C,saved['target_cross'],rtol=1e-10,atol=1e-10);np.testing.assert_allclose(E,saved['target_energy'],rtol=1e-10)
 for rep in ['source','hidden']:
  GG=T.T@G@T if rep=='source' else G;CC=T.T@C if rep=='source' else C;XX=X@T if rep=='source' else X
  norm=np.sqrt(np.diag(GG));GN=GG/norm[:,None]/norm[None,:];CN=CC/norm[:,None];Z=XX/norm
  eig,V=eigh((GN+GN.T)/2,check_finite=False);maxeig=max(eig);assert min(eig)>-1e-8*maxeig
  floors=[]
  for cutoff in [1e-10,1e-12]:
   keep=eig>cutoff*maxeig;proj=V[:,keep].T@CN;floor=(E-np.sum(proj**2/eig[keep,None],axis=0))/E
   assert min(floor)>-1e-6
   floors.append(dict(cutoff=cutoff,rank=int(sum(keep)),relative_floor=floor.tolist()))
  diagnostics.append(dict(key=key,model=model,representation=rep,features=len(norm),floor_checks=floors,min_normalized_gram_eigenvalue=float(min(eig)),max_normalized_gram_eigenvalue=float(maxeig)))
  for n in [32,128]:
   coef=saved[f'{rep}_{n}'];b=coef*norm[:,None]
   # Independent SVD ridge, rather than producer primal/dual normal equations.
   U,s,Vh=np.linalg.svd(Z[:n],full_matrices=False);ref=Vh.T@((s/(s*s+n*.001))[:,None]*(U.T@y[:n]))
   relative=np.linalg.norm(Z[:n]@(b-ref))/max(1.,np.linalg.norm(Z[:n]@ref));assert relative<1e-7
   err=np.sum(b*(GN@b),axis=0)-2*np.sum(b*CN,axis=0)+E
   old=next(r for r in rr if r['representation']==rep and r['n']==n)
   np.testing.assert_allclose(err/E,old['relative_mse'],rtol=1e-7,atol=1e-8)
   records.append(dict(key=key,model=model,representation=rep,n=n,svd_prediction_relative_error=float(relative)))
 print(key,model,'audited',flush=True)
(root/'audit.json').write_text(json.dumps(dict(status='passed',fits=len(records),records=records),indent=2)+'\n')
(root/'population_floors.json').write_text(json.dumps(diagnostics,indent=2)+'\n')

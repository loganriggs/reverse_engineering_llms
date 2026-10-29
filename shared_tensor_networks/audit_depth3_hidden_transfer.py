import hashlib,json
from pathlib import Path
import numpy as np
import torch
from directional_sampling import DirectionalMixture
from depth3_hidden_transfer import numpy_features
root=Path('results/depth3_hidden_transfer');src=Path('results/scaling_grid');rows=json.loads((root/'results.json').read_text());assert len(rows)==48
records=[]
def draw(axes,d,n,seed):
 u,w=DirectionalMixture(axes,(16,64)).sample(n,np.random.default_rng(seed));return u*np.sqrt(d),w
for row in rows:
 key,name,d,base=row['key'],row['model'],row['d'],row['base_seed'];folder=src/key;file=folder/('teacher.pt' if name=='oracle_source' else name+'.pt')
 assert hashlib.sha256(file.read_bytes()).hexdigest()==row['checkpoint_sha256'];assert hashlib.sha256((folder/'training.npz').read_bytes()).hexdigest()==row['data_sha256']
 state={k:v.numpy() for k,v in torch.load(file,weights_only=True).items()};teacher={k:v.numpy() for k,v in torch.load(folder/'teacher.pt',weights_only=True).items()};data=np.load(folder/'training.npz');save=np.load(root/f'{key}_{name}.npz')
 moments=[]
 for batch in range(4):
  x,w=draw(save['calibration_axes'],d,4096,base+batch);f,p=numpy_features(state,x);v=np.concatenate([p[:,:12],f],axis=1);moments.append(np.mean(v*v*w[:,None],axis=0))
 moments=np.array(moments);np.testing.assert_allclose(moments,save['calibration_moments'],rtol=1e-8,atol=1e-10);norm=np.sqrt(np.maximum(moments.mean(axis=0),1e-24));np.testing.assert_allclose(norm,save['norm'],rtol=1e-8,atol=1e-10)
 f,p=numpy_features(state,data['adaptation_inputs']);_,target=numpy_features(teacher,data['adaptation_inputs']);np.testing.assert_allclose(target[:,12:],data['adaptation_targets'],rtol=1e-8,atol=1e-10)
 Z=np.concatenate([p[:,:12],f],axis=1)/norm
 svderr=[]
 for i,(lo,hi,n) in enumerate([(0,12,32),(0,12,128),(12,len(norm),32),(12,len(norm),128)]):
  U,s,Vh=np.linalg.svd(Z[:n,lo:hi],full_matrices=False);b=Vh.T@((s/(s*s+n*.001))[:,None]*(U.T@data['adaptation_targets'][:n]))
  e=np.linalg.norm(Z[:,lo:hi]@(b-save[f'coef_{i}']))/max(1.,np.linalg.norm(Z[:,lo:hi]@b));assert e<1e-7;svderr.append(float(e))
 # Degree-eight homogeneity and independence from held-out source slots.
 probe=data['adaptation_inputs'][:8];f1,p1=numpy_features(state,probe);f2,p2=numpy_features(state,probe*1.2)
 np.testing.assert_allclose(f2,f1*1.2**8,rtol=1e-8,atol=1e-10);np.testing.assert_allclose(p2,p1*1.2**8,rtol=1e-8,atol=1e-10)
 if name=='oracle_source':
  changed={k:v.copy() for k,v in state.items()};changed['W'][12:]*=3
  if changed['A.0'].ndim==3:
   for k in ['A.0','B.0','A.1','B.1']:changed[k][12:]*=2
  ff,pp=numpy_features(changed,probe);np.testing.assert_array_equal(ff,f1);np.testing.assert_array_equal(pp[:,:12],p1[:,:12])
 batches=[]
 for batch in range(8):
  x,w=draw(save['evaluation_axes'],d,8192,base+200+batch);f,p=numpy_features(state,x);f=np.concatenate([p[:,:12],f],axis=1)/norm
  pred=np.concatenate([f[:,:12]@save['coef_0'],f[:,:12]@save['coef_1'],f[:,12:]@save['coef_2'],f[:,12:]@save['coef_3']],axis=1)
  _,target=numpy_features(teacher,x);y=np.tile(target[:,12:],(1,4));batches.append(np.mean(np.concatenate([(pred-y)**2,y*y],axis=1)*w[:,None],axis=0))
 batches=np.array(batches);np.testing.assert_allclose(batches,save['batches'],rtol=1e-8,atol=1e-10)
 mean=batches.mean(axis=0);rat=mean[:16]/mean[16:];cv=batches.std(axis=0,ddof=1)/np.maximum(abs(mean),1e-300);se=(batches[:,:16]-rat*batches[:,16:]).std(axis=0,ddof=1)/np.sqrt(8)/mean[16:]
 np.testing.assert_allclose(rat.reshape(4,4),row['relative_mse'],rtol=1e-8);np.testing.assert_allclose(se.reshape(4,4),row['ratio_standard_error'],rtol=1e-8,atol=1e-12)
 assert np.array_equal(((cv[:16]<.1)&(cv[16:]<.1)).reshape(4,4),row['precision_pass'])
 records.append(dict(key=key,model=name,max_svd_prediction_error=max(svderr),max_relative_batch_replay_error=float(np.max(abs(batches-save['batches'])/np.maximum(abs(save['batches']),1e-20)))))
 print(key,name,'audited',flush=True)
(root/'audit.json').write_text(json.dumps(dict(status='passed',models=48,adaptation_fits=192,evaluation_batches=384,records=records),indent=2)+'\n')

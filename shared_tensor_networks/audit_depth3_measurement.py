import hashlib,json
from pathlib import Path
import numpy as np
import torch
from directional_sampling import DirectionalMixture
from depth3_hidden_transfer import numpy_features
root=Path('results/depth3_measurement');prior=Path('results/depth3_hidden_transfer');src=Path('results/scaling_grid');rows=json.loads((root/'results.json').read_text());assert len(rows)==12
records=[]
def draw(axes,n,seed):
 u,w=DirectionalMixture(axes,(16,64)).sample(n,np.random.default_rng(seed));return u*np.sqrt(32),w
for row in rows:
 key,name,base=row['key'],row['model'],row['base_seed'];folder=src/key;file=folder/('teacher.pt' if name=='oracle_source' else name+'.pt');oldfile=prior/f'{key}_{name}.npz'
 assert hashlib.sha256(file.read_bytes()).hexdigest()==row['checkpoint_sha256'];assert hashlib.sha256(oldfile.read_bytes()).hexdigest()==row['old_fit_sha256']
 state={k:v.numpy() for k,v in torch.load(file,weights_only=True).items()};teacher={k:v.numpy() for k,v in torch.load(folder/'teacher.pt',weights_only=True).items()};data=np.load(folder/'training.npz');save=np.load(root/f'{key}_{name}.npz');old=np.load(oldfile)
 np.testing.assert_array_equal(save['norms'][0],old['norm']);np.testing.assert_array_equal(save['coef_0'],old['coef_1']);np.testing.assert_array_equal(save['coef_1'],old['coef_3'])
 norms=[old['norm']]
 for cal in range(2):
  moments=[]
  for b in range(4):
   x,w=draw(old['calibration_axes'],4096,base+cal*10+b);f,p=numpy_features(state,x);v=np.concatenate([p[:,:12],f],axis=1);moments.append(np.mean(v*v*w[:,None],axis=0))
  moments=np.array(moments);np.testing.assert_allclose(moments,save['calibration_moments'][cal],rtol=1e-8,atol=1e-10);norms.append(np.sqrt(np.maximum(moments.mean(axis=0),1e-24)))
 np.testing.assert_allclose(norms,save['norms'],rtol=1e-8,atol=1e-10)
 f,p=numpy_features(state,data['adaptation_inputs']);X=np.concatenate([p[:,:12],f],axis=1);mapping=[];max_svd=0.
 for cal,norm in enumerate(norms):
  for typ,sl in enumerate([slice(0,12),slice(12,None)]):
   i=cal*2+typ;Z=X[:,sl]/norm[sl];U,s,Vh=np.linalg.svd(Z,full_matrices=False);coef=Vh.T@((s/(s*s+128*.001))[:,None]*(U.T@data['adaptation_targets']))
   error=np.linalg.norm(Z@(coef-save[f'coef_{i}']))/max(1.,np.linalg.norm(Z@coef));assert error<1e-7;max_svd=max(max_svd,error)
   m=np.zeros((len(norm),4));m[sl]=save[f'coef_{i}']/norm[sl,None];mapping.append(m)
 mapping=np.concatenate(mapping,axis=1);np.testing.assert_allclose(mapping,save['mapping'],rtol=1e-8,atol=1e-10)
 panels=[]
 for panel in range(2):
  batches=[]
  for b in range(4):
   x,w=draw(save['axes'][panel],8192,base+200+panel*10+b);f,p=numpy_features(state,x);pred=np.concatenate([p[:,:12],f],axis=1)@mapping;_,yt=numpy_features(teacher,x);y=np.tile(yt[:,12:],(1,6));v=np.concatenate([(pred-y)**2,y*y],axis=1)*w[:,None];batches.append(v.mean(axis=0))
  panels.append(batches)
 panels=np.array(panels);np.testing.assert_allclose(panels,save['batches'],rtol=1e-8,atol=1e-10)
 flat=panels.reshape(8,48);mean=flat.mean(axis=0);ratio=mean[:24]/mean[24:];cv=flat.std(axis=0,ddof=1)/np.maximum(abs(mean),1e-300);se=(flat[:,:24]-ratio*flat[:,24:]).std(axis=0,ddof=1)/np.sqrt(8)/mean[24:]
 np.testing.assert_allclose(ratio.reshape(3,2,4),row['relative_mse'],rtol=1e-8);np.testing.assert_allclose(se.reshape(3,2,4),row['ratio_se'],rtol=1e-8,atol=1e-12);assert np.array_equal(((cv[:24]<.1)&(cv[24:]<.1)).reshape(3,2,4),row['precision_pass'])
 records.append(dict(key=key,model=name,max_svd_error=float(max_svd),max_relative_batch_replay_error=float(np.max(abs(panels-save['batches'])/np.maximum(abs(save['batches']),1e-20)))))
 print(key,name,'audited',flush=True)
(root/'audit.json').write_text(json.dumps(dict(status='passed',cases=12,readouts=72,new_readouts=48,evaluation_batches=96,records=records),indent=2)+'\n')

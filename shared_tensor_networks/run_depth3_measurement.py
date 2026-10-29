import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from depth3_hidden_transfer import load_model,features
from directional_sampling import DirectionalMixture
from run_scaling_grid import draws
from larger_hidden_transfer import ridge
src=Path('results/scaling_grid');previous=Path('results/depth3_hidden_transfer');out=Path('results/depth3_measurement');out.mkdir(exist_ok=False);rows=[];torch.set_num_threads(1)
for rep in range(3):
 key=f'g3_independent_r{rep}';folder=src/key;teacher=load_model(folder/'teacher.pt');data=np.load(folder/'training.npz');xa=torch.tensor(data['adaptation_inputs']);ya=data['adaptation_targets']
 for mi,name in enumerate(['shared','separate','shared_budget','oracle_source']):
  start=time.perf_counter();base=23800000+100000*rep+1000*mi;file=folder/('teacher.pt' if name=='oracle_source' else name+'.pt');net=load_model(file);old=np.load(previous/f'{key}_{name}.npz');norms=[old['norm']];moments=[];q=DirectionalMixture(old['calibration_axes'],(16,64))
  for cal in range(2):
   batches=[]
   for b in range(4):
    x,w=draws(q,32,4096,base+cal*10+b)
    with torch.no_grad():v=torch.cat([net(x)[:,:12],features(net,x)],dim=1);batches.append((v*v*w[:,None]).mean(dim=0).numpy())
   batches=np.array(batches);moments.append(batches);norms.append(np.sqrt(np.maximum(batches.mean(axis=0),1e-24)))
  with torch.no_grad():adapt=torch.cat([net(xa)[:,:12],features(net,xa)],dim=1).numpy()
  coefs=[old['coef_1'],old['coef_3']]
  for norm in norms[1:]:
   z=adapt/norm;coefs.extend([ridge(z[:,:12],ya),ridge(z[:,12:],ya)])
  # Combine coefficients in original feature coordinates to evaluate all six predictors.
  maps=[]
  for cal,norm in enumerate(norms):
   for feature in range(2):
    mat=np.zeros((len(norm),4));sl=slice(0,12) if feature==0 else slice(12,None);mat[sl]=coefs[2*cal+feature]/norm[sl,None];maps.append(mat)
  mapping=torch.tensor(np.concatenate(maps,axis=1))
  def predict(x):return torch.cat([net(x)[:,:12],features(net,x)],dim=1)@mapping
  all_axes=[];evals=[]
  for panel in range(2):
   rng=np.random.default_rng(base+100+panel);x=torch.tensor(rng.normal(size=(256,32)),requires_grad=True)
   target_indices=torch.arange(4).repeat_interleave(16);residual_indices=torch.arange(24).repeat_interleave(8)
   for step in range(100):
    u=x/torch.linalg.vector_norm(x,dim=1,keepdim=True)*np.sqrt(32);y=teacher(u)[:,12:];pred=predict(u);errors=pred-y.repeat(1,6)
    selected=torch.cat([y[:64].gather(1,target_indices[:,None]).ravel(),errors[64:].gather(1,residual_indices[:,None]).ravel()]);score=torch.log(selected.square()+1e-30).sum();grad,=torch.autograd.grad(score,x)
    with torch.no_grad():x+=.15*grad/torch.linalg.vector_norm(grad,dim=1,keepdim=True).clamp_min(1e-20);x/=torch.linalg.vector_norm(x,dim=1,keepdim=True)
   axes=x.detach().numpy();all_axes.append(axes);eq=DirectionalMixture(axes,(16,64));batches=[]
   for b in range(4):
    xx,w=draws(eq,32,8192,base+200+panel*10+b)
    with torch.no_grad():y=teacher(xx)[:,12:].repeat(1,6);pred=predict(xx);v=torch.cat([(pred-y).square(),y.square()],dim=1)*w[:,None];batches.append(v.mean(dim=0).numpy())
   evals.append(batches)
  evals=np.array(evals);flat=evals.reshape(8,48);mean=flat.mean(axis=0);ratio=mean[:24]/mean[24:];cv=flat.std(axis=0,ddof=1)/np.maximum(abs(mean),1e-300)
  se=(flat[:,:24]-ratio*flat[:,24:]).std(axis=0,ddof=1)/np.sqrt(8)/mean[24:]
  save=dict(norms=np.array(norms),calibration_moments=np.array(moments),axes=np.array(all_axes),batches=evals,mapping=mapping.numpy())
  for i,c in enumerate(coefs):save[f'coef_{i}']=c
  np.savez_compressed(out/f'{key}_{name}.npz',**save)
  row=dict(key=key,model=name,base_seed=base,relative_mse=ratio.reshape(3,2,4).tolist(),ratio_se=se.reshape(3,2,4).tolist(),precision_pass=((cv[:24]<.1)&(cv[24:]<.1)).reshape(3,2,4).tolist(),seconds=time.perf_counter()-start,checkpoint_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),old_fit_sha256=hashlib.sha256((previous/f'{key}_{name}.npz').read_bytes()).hexdigest())
  rows.append(row);(out/'results.json').write_text(json.dumps(rows,indent=2)+'\n');print(key,name,'seconds',round(row['seconds'],1),'passes',np.sum(row['precision_pass']),'/24',flush=True)

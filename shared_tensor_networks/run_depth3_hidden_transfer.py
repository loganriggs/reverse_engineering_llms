import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from depth3_hidden_transfer import load_model,features
from directional_sampling import DirectionalMixture
from run_scaling_grid import draws,proposal
from larger_hidden_transfer import ridge
root=Path('results/depth3_hidden_transfer');root.mkdir(exist_ok=False);src=Path('results/scaling_grid');rows=[];torch.set_num_threads(1)
for gi,d in [(1,16),(3,32)]:
 for ki,kind in enumerate(['shared','independent']):
  for rep in range(3):
   key=f'g{gi}_{kind}_r{rep}';folder=src/key;teacher=load_model(folder/'teacher.pt');data=np.load(folder/'training.npz');xa=torch.tensor(data['adaptation_inputs']);ya=data['adaptation_targets']
   group=(gi//2)*6+ki*3+rep
   for mi,name in enumerate(['shared','separate','shared_budget','oracle_source']):
    start=time.perf_counter();base=23600000+100000*group+1000*mi;file=folder/('teacher.pt' if name=='oracle_source' else name+'.pt');net=load_model(file)
    axes=np.load(folder/('teacher_calibration.npz' if name=='oracle_source' else name+'_evaluation.npz'))['axes' if name=='oracle_source' else 'feature_axes'];q=DirectionalMixture(axes,(16,64))
    calibration=[]
    for batch in range(4):
     x,w=draws(q,d,4096,base+batch)
     with torch.no_grad():v=torch.cat([net(x)[:,:12],features(net,x)],dim=1);calibration.append((v.square()*w[:,None]).mean(dim=0).numpy())
    calibration=np.array(calibration);norm=np.sqrt(np.maximum(calibration.mean(axis=0),1e-24));cv=calibration.std(axis=0,ddof=1)/np.maximum(calibration.mean(axis=0),1e-300)
    with torch.no_grad():adapt=torch.cat([net(xa)[:,:12],features(net,xa)],dim=1).numpy()/norm
    coefs=[]
    for lo,hi in [(0,12),(12,len(norm))]:
     for n in [32,128]:coefs.append(ridge(adapt[:n,lo:hi],ya[:n]))
    def target(x):return teacher(x)[:,12:].repeat(1,4)
    def predict(x):
     f=torch.cat([net(x)[:,:12],features(net,x)],dim=1)/torch.tensor(norm)
     return torch.cat([f[:,:12]@torch.tensor(coefs[0]),f[:,:12]@torch.tensor(coefs[1]),f[:,12:]@torch.tensor(coefs[2]),f[:,12:]@torch.tensor(coefs[3])],dim=1)
    eq,eaxes=proposal(target,predict,d,base+100);batches=[]
    for batch in range(8):
     x,w=draws(eq,d,8192,base+200+batch)
     with torch.no_grad():y=target(x);p=predict(x);values=torch.cat([(p-y).square(),y.square()],dim=1)*w[:,None];batches.append(values.mean(dim=0).numpy())
    batches=np.array(batches);means=batches.mean(axis=0);ratio=means[:16]/means[16:];bcv=batches.std(axis=0,ddof=1)/np.maximum(abs(means),1e-300)
    se=(batches[:,:16]-ratio*batches[:,16:]).std(axis=0,ddof=1)/np.sqrt(8)/means[16:]
    save=dict(calibration_axes=axes,calibration_moments=calibration,norm=norm,evaluation_axes=eaxes,batches=batches)
    for i,c in enumerate(coefs):save[f'coef_{i}']=c
    np.savez_compressed(root/f'{key}_{name}.npz',**save)
    row=dict(key=key,d=d,teacher=kind,model=name,base_seed=base,features=len(norm)-12,relative_mse=ratio.reshape(4,4).tolist(),ratio_standard_error=se.reshape(4,4).tolist(),precision_pass=((bcv[:16]<.1)&(bcv[16:]<.1)).reshape(4,4).tolist(),calibration_bad_fraction=float(np.mean(cv>.1)),seconds=time.perf_counter()-start,checkpoint_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),data_sha256=hashlib.sha256((folder/'training.npz').read_bytes()).hexdigest())
    rows.append(row);(root/'results.json').write_text(json.dumps(rows,indent=2)+'\n');print(key,name,'seconds',round(row['seconds'],1),'precision',sum(np.ravel(row['precision_pass'])),'/16',flush=True)

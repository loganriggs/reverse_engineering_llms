"""Teacher-initialized coefficient-noise sensitivity, invariant feature metric."""
import time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from run_readout_recovery_h3 import maps,quadratic,L,L2,N,H,P
from run_svhn_standard import digest,write

def features(v):
 q=L2@quadratic(v[:N].reshape(H,-1),v[N:2*N].reshape(H,-1))
 return (q/np.linalg.norm(q,axis=0)/np.sqrt(H)).ravel()

def feature_jacobian(v):
 q=L2@quadratic(v[:N].reshape(H,-1),v[N:2*N].reshape(H,-1));result=np.zeros((q.size,len(v)))
 for i in range(2*N):
  vp=v.copy();vm=v.copy();vp[i]+=1;vm[i]-=1
  dq=L2@(quadratic(vp[:N].reshape(H,-1),vp[N:2*N].reshape(H,-1))-quadratic(vm[:N].reshape(H,-1),vm[N:2*N].reshape(H,-1)))/2
  unit=q/np.linalg.norm(q,axis=0);derivative=(dq-unit*np.sum(unit*dq,axis=0))/np.linalg.norm(q,axis=0)/np.sqrt(H)
  result[:,i]=derivative.ravel()
 return result

def main():
 root=Path(__file__).parent;out=root/'results/readout_stability';out.mkdir(exist_ok=False)
 protocol=dict(teacher_seeds=[218001,218002,218003],arms=['one','three','six','interventions'],amplitudes=[1e-5,1e-3],directions=['random','worst'],max_nfev=100,
  noise='unit vector in whitened coefficient observation space, after per-head RMS normalization and averaging over heads; fixed total RMS amplitude, not iid sample-label noise',
  initialization='exact teacher parameters; privileged local stability test, not learning from random initialization',feature_metric='mean squared change in unit-Gaussian-norm feature functions, fixed local component ordering',
  source_sha256=digest(Path(__file__)),dependencies={s:digest(root/s) for s in ['run_readout_recovery_h3.py','experiment.py','multitask.py']})
 write(out/'protocol.json',protocol);(out/'source.py').write_bytes(Path(__file__).read_bytes())
 for seed in protocol['teacher_seeds']:
  for arm in protocol['arms']:
   source=root/f'results/readout_recovery_h3/{seed}_219001_{arm}.npz';d=np.load(source);k={'one':1,'three':3,'six':6,'interventions':1}[arm]
   v0=np.concatenate([d['teacher_A'].ravel(),d['teacher_B'].ravel(),d['teacher_W'][:k].ravel()]);scales=d['scales'];m=len(scales)
   def output(v):return (L@(maps(v,arm)[0]/scales)/np.sqrt(m)).ravel()
   def jac(v):return (np.einsum('ij,jtp->itp',L,maps(v,arm)[1])/scales[None,:,None]/np.sqrt(m)).reshape(-1,len(v))
   target=output(v0);J=jac(v0);F=feature_jacobian(v0);u,s,vh=np.linalg.svd(J,full_matrices=False);rank=int((s>s[0]*1e-10).sum());assert rank==len(v0)-6
   inverse=(vh[:rank].T/s[:rank])@u[:,:rank].T;response=F@inverse
   null_error=float(np.linalg.norm(F@vh[rank:].T));assert null_error<1e-8
   # Coordinate-invariant response: changing parameter units must not change this map.
   coordinate=np.exp(np.linspace(-1,1,len(v0)));transformed=(F*coordinate)@np.linalg.pinv(J*coordinate,rcond=1e-10)
   coordinate_error=float(np.linalg.norm(response-transformed)/max(1,np.linalg.norm(response)));assert coordinate_error<1e-7
   _,gain,right=np.linalg.svd(response,full_matrices=False)
   random=np.random.default_rng(seed+4000).normal(size=(126,6))[:,:m].ravel();random/=np.linalg.norm(random)
   fixture=f'{seed}_{arm}';np.savez_compressed(out/f'{fixture}_geometry.npz',teacher=v0,scales=scales,J=J,F=F,response=response,singular_values=s)
   write(out/f'{fixture}_geometry.json',dict(name=fixture,rank=rank,parameters=len(v0),worst_feature_gain=float(gain[0]),random_direction_rms_gain=float(np.linalg.norm(response)/np.sqrt(len(target))),null_feature_error=null_error,coordinate_error=coordinate_error,input_file=str(source.relative_to(root)),input_sha256=digest(source)))
   for direction,eta in [('random',random),('worst',right[0])]:
    for amplitude in protocol['amplitudes']:
     noisy=target+amplitude*eta
     fun=lambda v:output(v)-noisy
     start=time.monotonic();fit=least_squares(fun,v0,jac=jac,method='trf',max_nfev=100,ftol=1e-12,xtol=1e-12,gtol=1e-12)
     predicted=amplitude*response@eta;actual=features(fit.x)-features(v0)
     name=f'{fixture}_{direction}_{amplitude}'
     np.savez_compressed(out/f'{name}.npz',initial=v0,final=fit.x,noise=eta,actual_feature_change=actual,predicted_feature_change=predicted)
     write(out/f'{name}.json',dict(name=name,teacher_seed=seed,arm=arm,direction=direction,amplitude=amplitude,nfev=fit.nfev,njev=fit.njev,status=int(fit.status),message=fit.message,optimality=float(fit.optimality),seconds=time.monotonic()-start,
       feature_change=float(np.linalg.norm(actual)),predicted_change=float(np.linalg.norm(predicted)),relative_linearization_error=float(np.linalg.norm(actual-predicted)/max(np.linalg.norm(predicted),1e-30)),noisy_fit_mse=float(fun(fit.x)@fun(fit.x)),clean_output_mse=float(np.sum((output(fit.x)-target)**2)),checkpoint_sha256=digest(out/f'{name}.npz')))
     print(name,float(np.linalg.norm(actual)),float(np.linalg.norm(actual-predicted)/max(np.linalg.norm(predicted),1e-30)),flush=True)
 write(out/'completed.json',dict(status='complete',fits=48,geometries=12))
if __name__=='__main__':main()

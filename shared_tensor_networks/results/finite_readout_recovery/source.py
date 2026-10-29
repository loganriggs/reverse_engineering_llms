"""Fixed scalar-label budgets, noisy finite samples, held-out task transfer."""
import time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares,linear_sum_assignment
from run_readout_recovery_h3 import maps,design,KEYS,L,L2,quadratic,N,H,P
from multitask import product_map
from run_svhn_standard import digest,write

def main():
 root=Path(__file__).parent;out=root/'results/finite_readout_recovery';out.mkdir(exist_ok=False)
 protocol=dict(teacher_seeds=[218001,218002,218003],budgets=[384,1536],heads=[1,3,6],noise_std=[0.,.05],initializations=['random','teacher'],max_nfev=200,
  pairing='one fixed nested 1536-input Gaussian pool and 1536x6 iid Gaussian noise array per teacher; head prefixes, n=scalar_budget/heads',
  supervision='fixed total source scalar-label budget; normalization uses known teacher Gaussian RMS, a privileged simulation convention shared by all arms',
  transfer='four whole heads held out; frozen hidden products,32/128new labels per head; paired Gaussian noise; oracle true products retained',
  optimizer='SciPyTRF analytic Jacobian,ftol/xtol/gtol1e-12,no restarts, no validation-based selection',
  source_sha256=digest(Path(__file__)),dependencies={s:digest(root/s) for s in ['run_readout_recovery_h3.py','experiment.py','multitask.py']})
 write(out/'protocol.json',protocol);(out/'source.py').write_bytes(Path(__file__).read_bytes())
 for seed in protocol['teacher_seeds']:
  source=root/f'results/readout_recovery_h3/{seed}_219001_six.npz';d=np.load(source);teacherK=product_map(d['teacher_A'],d['teacher_B'],KEYS)
  full=teacherK@d['teacher_W'].T;norms=np.linalg.norm(L@full,axis=0);truth=full/norms
  rng=np.random.default_rng(seed+5000);x=rng.normal(size=(1536,6));noise=rng.normal(size=(1536,6));adapt=rng.normal(size=(128,6));adapt_noise=rng.normal(size=(128,4));phi=design(x,KEYS);pa=design(adapt,KEYS)
  np.savez_compressed(out/f'{seed}_data.npz',inputs=x,noise=noise,adaptation_inputs=adapt,adaptation_noise=adapt_noise,truth=truth,teacher_A=d['teacher_A'],teacher_B=d['teacher_B'],teacher_W=d['teacher_W'],norms=norms)
  write(out/f'{seed}_data.json',dict(input_file=str(source.relative_to(root)),input_sha256=digest(source),data_sha256=digest(out/f'{seed}_data.npz')))
  for budget in protocol['budgets']:
   for m in protocol['heads']:
    n=budget//m;X=phi[:n];scales=norms[:m]
    for sigma in protocol['noise_std']:
     y=X@truth[:,:m]+sigma*noise[:n,:m]
     baseline=np.linalg.lstsq(X,y,rcond=1e-12)[0];baseline_mse=np.sum((L@(baseline-truth[:,:m]))**2,axis=0)
     for init in protocol['initializations']:
      W=d['teacher_W'][:m] if init=='teacher' else d['initial'][36:].reshape(6,6)[:m]
      v0=np.concatenate([d['teacher_A'].ravel(),d['teacher_B'].ravel(),W.ravel()]) if init=='teacher' else np.concatenate([d['initial'][:36],W.ravel()])
      def fun(v):return ((X@(maps(v,'six')[0]/scales)-y)/np.sqrt(budget)).ravel()
      def jac(v):return (np.einsum('ij,jtp->itp',X,maps(v,'six')[1])/scales[None,:,None]/np.sqrt(budget)).reshape(-1,len(v))
      start=time.monotonic();fit=least_squares(fun,v0,jac=jac,method='trf',max_nfev=200,ftol=1e-12,xtol=1e-12,gtol=1e-12)
      C,_,K,Q=maps(fit.x,'six');delta=C/scales-truth[:,:m];population=np.sum((L@delta)**2,axis=0)
      tq=L2@quadratic(d['teacher_A'],d['teacher_B']);sq=L2@Q;tq/=np.linalg.norm(tq,axis=0);sq/=np.linalg.norm(sq,axis=0)
      similarity=np.abs(tq.T@sq);ii,jj=linear_sum_assignment(-similarity)
      transfer={};saved={}
      for labelcount in [32,128]:
       Y=pa[:labelcount]@truth[:,6:]+sigma*adapt_noise[:labelcount]
       for feature,Kf in [('learned',K),('oracle',teacherK)]:
        Z=pa[:labelcount]@Kf;readout=np.linalg.lstsq(Z,Y,rcond=1e-12)[0];diff=Kf@readout-truth[:,6:];key=f'{feature}_{labelcount}'
        transfer[key]=np.sum((L@diff)**2,axis=0).tolist();saved['readout_'+key]=readout
      name=f'{seed}_b{budget}_h{m}_noise{sigma}_{init}'
      np.savez_compressed(out/f'{name}.npz',initial=v0,final=fit.x,polynomial_baseline=baseline,**saved)
      write(out/f'{name}.json',dict(name=name,teacher_seed=seed,budget=budget,heads=m,ninputs=n,noise_std=sigma,initialization=init,parameters=len(v0),train_mse=float(fun(fit.x)@fun(fit.x)),population_mse=population.tolist(),polynomial_population_mse=baseline_mse.tolist(),component_cosines=similarity[ii,jj].tolist(),transfer_mse=transfer,nfev=fit.nfev,njev=fit.njev,status=int(fit.status),message=fit.message,optimality=float(fit.optimality),seconds=time.monotonic()-start,checkpoint_sha256=digest(out/f'{name}.npz')))
      print(name,float(population.mean()),float(similarity[ii,jj].min()),flush=True)
 write(out/'completed.json',dict(status='complete',fits=72))
if __name__=='__main__':main()

"""Fresh-teacher confirmation of fixed-budget noisy task diversity."""
import time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares,linear_sum_assignment
import itertools
from experiment import gram
from multitask import monomials
from direct_bilinear_readout import predict_jacobian,quadratic_coefficients as quadratic
D=8;H=4;N=D*H;P=H*(H+1)//2
KEYS=monomials(D);L=np.linalg.cholesky(gram(KEYS)).T
QKEYS=[tuple(pair.count(i) for i in range(D)) for pair in itertools.combinations_with_replacement(range(D),2)]
L2=np.linalg.cholesky(gram(QKEYS)).T
def design(x,keys):return np.prod(x[:,None,:]**np.array(keys)[None,:,:],axis=2)
from multitask import product_map
from run_svhn_standard import digest,write

def main():
 root=Path(__file__).parent;out=root/'results/finite_readout_h4';out.mkdir(exist_ok=False)
 protocol=dict(teacher_seeds=[225001,225002,225003,225004,225005],budgets=[1280,5120],heads=[1,4,10],noise_std=[.05],initializations=['random226001','random226002','random226003','teacher'],max_nfev=200,
  pairing='one fixed nested 5120-input Gaussian pool and 5120x10 iid Gaussian noise array per teacher; head prefixes, n=scalar_budget/heads',
  supervision='fixed total source scalar-label budget; normalization uses known teacher Gaussian RMS, a privileged simulation convention shared by all arms',
  transfer='four whole heads held out; frozen hidden products,32/128new labels per head; paired Gaussian noise; oracle true products retained',
  optimizer='SciPyTRF analytic Jacobian,ftol/xtol/gtol1e-12,no restarts, no validation-based selection',
  source_sha256=digest(Path(__file__)),dependencies={s:digest(root/s) for s in ['direct_bilinear_readout.py','experiment.py','multitask.py']})
 write(out/'protocol.json',protocol);(out/'source.py').write_bytes(Path(__file__).read_bytes())
 for seed in protocol['teacher_seeds']:
  teacher_rng=np.random.default_rng(seed)
  d=dict(teacher_A=teacher_rng.normal(size=(H,D))/np.sqrt(D),teacher_B=teacher_rng.normal(size=(H,D))/np.sqrt(D),teacher_W=teacher_rng.normal(size=(P+4,P))/np.sqrt(P))
  assert np.linalg.matrix_rank(np.concatenate([d['teacher_A'],d['teacher_B']]))==D
  teacherK=product_map(d['teacher_A'],d['teacher_B'],KEYS)
  full=teacherK@d['teacher_W'].T;norms=np.linalg.norm(L@full,axis=0);truth=full/norms
  rng=np.random.default_rng(seed+5000);x=rng.normal(size=(5120,D));noise=rng.normal(size=(5120,P));adapt=rng.normal(size=(128,D));adapt_noise=rng.normal(size=(128,4));phi=design(x,KEYS);pa=design(adapt,KEYS)
  np.savez_compressed(out/f'{seed}_data.npz',inputs=x,noise=noise,adaptation_inputs=adapt,adaptation_noise=adapt_noise,truth=truth,teacher_A=d['teacher_A'],teacher_B=d['teacher_B'],teacher_W=d['teacher_W'],norms=norms)
  write(out/f'{seed}_data.json',dict(teacher_seed=seed,data_sha256=digest(out/f'{seed}_data.npz')))
  for budget in protocol['budgets']:
   for m in protocol['heads']:
    n=budget//m;X=phi[:n];scales=norms[:m]
    for sigma in protocol['noise_std']:
     y=X@truth[:,:m]+sigma*noise[:n,:m]
     baseline=np.linalg.lstsq(X,y,rcond=1e-12)[0];baseline_mse=np.sum((L@(baseline-truth[:,:m]))**2,axis=0)
     for init in protocol['initializations']:
      if init!='teacher':
       irng=np.random.default_rng(int(init.removeprefix('random')))
       d['initial']=np.concatenate([(irng.normal(size=(H,D))/np.sqrt(D)).ravel(),(irng.normal(size=(H,D))/np.sqrt(D)).ravel(),(irng.normal(size=(P,P))/np.sqrt(P)).ravel()])
      W=d['teacher_W'][:m] if init=='teacher' else d['initial'][2*N:].reshape(P,P)[:m]
      v0=np.concatenate([d['teacher_A'].ravel(),d['teacher_B'].ravel(),W.ravel()]) if init=='teacher' else np.concatenate([d['initial'][:2*N],W.ravel()])
      def fun(v):return ((predict_jacobian(v,x[:n],H,False)/scales-y)/np.sqrt(budget)).ravel()
      def jac(v):return (predict_jacobian(v,x[:n],H)[1]/scales[None,:,None]/np.sqrt(budget)).reshape(-1,len(v))
      start=time.monotonic();fit=least_squares(fun,v0,jac=jac,method='trf',max_nfev=200,ftol=1e-12,xtol=1e-12,gtol=1e-12)
      A=fit.x[:N].reshape(H,D);B=fit.x[N:2*N].reshape(H,D);K=product_map(A,B,KEYS);Q=quadratic(A,B);C=K@fit.x[2*N:].reshape(m,P).T;delta=C/scales-truth[:,:m];population=np.sum((L@delta)**2,axis=0)
      tq=L2@quadratic(d['teacher_A'],d['teacher_B']);sq=L2@Q;tq/=np.linalg.norm(tq,axis=0);sq/=np.linalg.norm(sq,axis=0)
      similarity=np.abs(tq.T@sq);ii,jj=linear_sum_assignment(-similarity)
      transfer={};saved={}
      for labelcount in [32,128]:
       Y=pa[:labelcount]@truth[:,P:]+sigma*adapt_noise[:labelcount]
       for feature,Kf in [('learned',K),('oracle',teacherK)]:
        Z=pa[:labelcount]@Kf;readout=np.linalg.lstsq(Z,Y,rcond=1e-12)[0];diff=Kf@readout-truth[:,P:];key=f'{feature}_{labelcount}'
        transfer[key]=np.sum((L@diff)**2,axis=0).tolist();saved['readout_'+key]=readout
      name=f'{seed}_b{budget}_h{m}_noise{sigma}_{init}'
      np.savez_compressed(out/f'{name}.npz',initial=v0,final=fit.x,polynomial_baseline=baseline,**saved)
      write(out/f'{name}.json',dict(name=name,teacher_seed=seed,budget=budget,heads=m,ninputs=n,noise_std=sigma,initialization=init,parameters=len(v0),train_mse=float(fun(fit.x)@fun(fit.x)),population_mse=population.tolist(),polynomial_population_mse=baseline_mse.tolist(),component_cosines=similarity[ii,jj].tolist(),transfer_mse=transfer,nfev=fit.nfev,njev=fit.njev,status=int(fit.status),message=fit.message,optimality=float(fit.optimality),seconds=time.monotonic()-start,checkpoint_sha256=digest(out/f'{name}.npz')))
      print(name,float(population.mean()),float(similarity[ii,jj].min()),flush=True)
 write(out/'completed.json',dict(status='complete',fits=120))
if __name__=='__main__':main()

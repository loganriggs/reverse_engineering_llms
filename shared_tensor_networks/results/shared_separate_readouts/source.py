"""Shared versus independent heads; identical labels and ten output transfer features."""
import time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from experiment import gram
from multitask import monomials,product_map
from direct_bilinear_readout import predict_jacobian
from run_svhn_standard import write,digest
keys=monomials(8);L=np.linalg.cholesky(gram(keys)).T
def design(x):return np.prod(x[:,None,:]**np.array(keys)[None,:,:],axis=2)
def coefficient(v,h):
 n=8*h;p=h*(h+1)//2
 return product_map(v[:n].reshape(h,8),v[n:2*n].reshape(h,8),keys)@v[2*n:].reshape(-1,p).T

def main():
 root=Path(__file__).parent;out=root/'results/shared_separate_readouts';out.mkdir(exist_ok=False)
 write(out/'protocol.json',dict(teachers=list(range(225001,225006)),initializations=list(range(226001,226004)),budget=5120,heads=10,ninputs=512,noise=.05,max_nfev_per_head=200,source_sha256=digest(Path(__file__)),dependencies={s:digest(root/s) for s in ['direct_bilinear_readout.py','experiment.py','multitask.py']}));(out/'source.py').write_bytes(Path(__file__).read_bytes())
 for seed in range(225001,225006):
  datafile=root/f'results/finite_readout_h4/{seed}_data.npz';data=np.load(datafile);x=data['inputs'][:512];phi=design(x);pa=design(data['adaptation_inputs']);truth=data['truth'];norms=data['norms'];y=phi@truth[:,:10]+.05*data['noise'][:512,:10]
  for init in range(226001,226004):
   source=root/f'results/finite_readout_h4/{seed}_b5120_h10_noise0.05_random{init}.npz';shared=np.load(source);sv=shared['initial'];initialA=sv[:32].reshape(4,8);initialB=sv[32:64].reshape(4,8);initialW=sv[64:].reshape(10,10)
   for method in ['shared','separate_width','separate_budget']:
    name=f'{seed}_{init}_{method}';params={};headrecords=[];start=time.monotonic()
    if method=='shared':C=coefficient(shared['final'],4)/norms[:10];count=164
    else:
     h=4 if method=='separate_width' else 1;p=h*(h+1)//2;columns=[];count=10*(16*h+p)
     for t in range(10):
      v0=np.concatenate([initialA[:h].ravel(),initialB[:h].ravel(),initialW[t,:p]])
      fun=lambda v:(predict_jacobian(v,x,h,False)[:,0]/norms[t]-y[:,t])/np.sqrt(512)
      jac=lambda v:predict_jacobian(v,x,h)[1][:,0,:]/norms[t]/np.sqrt(512)
      if h==4:np.testing.assert_allclose(predict_jacobian(v0,x,h,False)[:,0],predict_jacobian(sv,x,4,False)[:,t],atol=1e-12,rtol=1e-12)
      fit=least_squares(fun,v0,jac=jac,method='trf',max_nfev=200,ftol=1e-12,xtol=1e-12,gtol=1e-12)
      params[f'initial_{t}']=v0;params[f'final_{t}']=fit.x;columns.append(coefficient(fit.x,h)[:,0]/norms[t]);headrecords.append(dict(head=t,nfev=fit.nfev,njev=fit.njev,status=int(fit.status),message=fit.message,train_mse=float(fun(fit.x)@fun(fit.x))))
     C=np.stack(columns,axis=1)
    delta=C-truth[:,:10];source_mse=np.sum((L@delta)**2,axis=0);norm=np.linalg.norm(L@C,axis=0);assert (norm>1e-12).all();features=C/norm;transfer={}
    for feature,F in [('learned',features),('oracle',truth[:,:10])]:
     for labels in ['population',32,128]:
      X,Y=(L@F,L@truth[:,10:]) if labels=='population' else (pa[:labels]@F,pa[:labels]@truth[:,10:]+.05*data['adaptation_noise'][:labels])
      readout=np.linalg.lstsq(X,Y,rcond=1e-12)[0];diff=F@readout-truth[:,10:];key=f'{feature}_{labels}';params['readout_'+key]=readout;transfer[key]=np.sum((L@diff)**2,axis=0).tolist()
    np.savez_compressed(out/f'{name}.npz',coefficients=C,features=features,**params)
    write(out/f'{name}.json',dict(name=name,teacher_seed=seed,initial_seed=init,method=method,parameters=count,source_mse=source_mse.tolist(),transfer_mse=transfer,head_records=headrecords,seconds=time.monotonic()-start,input_file=str(source.relative_to(root)),input_sha256=digest(source),data_file=str(datafile.relative_to(root)),data_sha256=digest(datafile),checkpoint_sha256=digest(out/f'{name}.npz')))
    print(name,float(source_mse.mean()),float(np.mean(transfer['learned_128'])),flush=True)
 write(out/'completed.json',dict(status='complete',groups=45,individual_head_fits=300))
if __name__=='__main__':main()

"""Exact-population information ablation for hidden quadratic recovery."""
import itertools,time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.optimize import linear_sum_assignment
from experiment import gram,multiply
from multitask import monomials,product_map
from run_svhn_standard import digest,write

D=6;H=3;PAIRS=list(itertools.combinations_with_replacement(range(H),2));P=len(PAIRS);N=H*D
MASKS=np.array([[1]*P]+[[int(i!=h and j!=h) for i,j in PAIRS] for h in range(H)])
QKEYS=list(itertools.combinations_with_replacement(range(D),2))
QEXP=[tuple(pair.count(i) for i in range(D)) for pair in QKEYS]
KEYS=monomials(D);G=gram(KEYS);L=np.linalg.cholesky(G).T
G2=gram(QEXP);L2=np.linalg.cholesky(G2).T
T=np.zeros((len(KEYS),len(QEXP),len(QEXP)))
for i,a in enumerate(QEXP):
 for j,b in enumerate(QEXP):T[KEYS.index(tuple(x+y for x,y in zip(a,b))),i,j]=1

def parts(v):return v[:N].reshape(H,D),v[N:2*N].reshape(H,D),v[2*N:].reshape(-1,P)
def quadratic(A,B):
 return np.stack([A[:,i]*B[:,j]+(A[:,j]*B[:,i] if i!=j else 0) for i,j in QKEYS])
def maps(v,arm):
 A,B,W=parts(v);Q=quadratic(A,B);dq=np.zeros((len(QKEYS),H,2*N))
 for t,(i,j) in enumerate(QKEYS):
  for h in range(H):
   dq[t,h,h*D+i]+=B[h,j];dq[t,h,N+h*D+j]+=A[h,i]
   if i!=j:dq[t,h,h*D+j]+=B[h,i];dq[t,h,N+h*D+i]+=A[h,j]
 K=np.stack([T.reshape(len(KEYS),-1)@np.outer(Q[:,i],Q[:,j]).ravel() for i,j in PAIRS],axis=1)
 dK=np.stack([T.reshape(len(KEYS),-1)@(dq[:,i,None,:]*Q[None,:,j,None]+Q[:,i,None,None]*dq[None,:,j,:]).reshape(len(QKEYS)**2,2*N) for i,j in PAIRS],axis=1)
 masks=MASKS if arm=='interventions' else None
 effective=W if masks is None else W*masks
 C=K@effective.T;J=np.zeros((len(KEYS),len(effective),len(v)));J[:,:,:2*N]=np.einsum('kip,ti->ktp',dK,effective)
 if masks is None:
  for t in range(len(W)):J[:,t,2*N+P*t:2*N+P*(t+1)]=K
 else:
  for t in range(H+1):J[:,t,2*N:]=K*masks[t]
 return C,J,K,Q

def design(x,keys):return np.prod(x[:,None,:]**np.array(keys)[None,:,:],axis=2)
def subspace(q):
 u,s,_=np.linalg.svd(q,full_matrices=False);return u[:,s>s[0]*1e-11]

def main():
 root=Path(__file__).parent;out=root/'results/readout_recovery_h3';out.mkdir(exist_ok=False)
 protocol=dict(d=D,h=H,teacher_seeds=[218001,218002,218003],initial_seeds=[219001,219002],arms=['one','three','six','interventions'],
  max_nfev=200,optimizer='scipy least_squares trf exact analytic Jacobian, linear loss, ftol/xtol/gtol1e-12',
  objective='mean normalized squared Gaussian coefficient error, no finite training samples',
  supervision='one/three/six independent heads; intervention arm base head plus individually ablate each hidden feature, known matched student indices',
  transfer='four independent heads withheld from representation fitting; population projection and 32/128 labeled least-squares adaptation on frozen hidden products',
  source_sha256=digest(Path(__file__)),dependencies={s:digest(root/s) for s in ['experiment.py','multitask.py']})
 write(out/'protocol.json',protocol);(out/'source.py').write_bytes(Path(__file__).read_bytes())
 for seed in protocol['teacher_seeds']:
  rng=np.random.default_rng(seed);A=rng.normal(size=(H,D))/np.sqrt(D);B=rng.normal(size=(H,D))/np.sqrt(D);W=rng.normal(size=(P+4,P))/np.sqrt(P)
  assert np.linalg.matrix_rank(np.concatenate([A,B]))==D
  K=product_map(A,B,KEYS);C=K@W.T;Q=quadratic(A,B)
  adapt=np.random.default_rng(seed+3000).normal(size=(128,D));Phi=design(adapt,KEYS)
  held=C[:,P:].copy();held/=np.linalg.norm(L@held,axis=0)
  for init in protocol['initial_seeds']:
   rngi=np.random.default_rng(init);ai=rngi.normal(size=(H,D))/np.sqrt(D);bi=rngi.normal(size=(H,D))/np.sqrt(D);wi=rngi.normal(size=(P,P))/np.sqrt(P)
   for arm in protocol['arms']:
    count={'one':1,'three':3,'six':P,'interventions':1}[arm]
    v0=np.concatenate([ai.ravel(),bi.ravel(),wi[:count].ravel()])
    truth=C[:,:count] if arm!='interventions' else K@(W[:1]*MASKS).T
    scales=np.linalg.norm(L@truth,axis=0);truth=truth/scales
    def fun(v):return ((L@(maps(v,arm)[0]/scales-truth))/np.sqrt(len(scales))).ravel()
    def jac(v):return (np.einsum('ij,jtp->itp',L,maps(v,arm)[1])/scales[None,:,None]/np.sqrt(len(scales))).reshape(-1,len(v))
    start=time.monotonic();fit=least_squares(fun,v0,jac=jac,method='trf',max_nfev=200,ftol=1e-12,xtol=1e-12,gtol=1e-12)
    fitted,_,studentK,studentQ=maps(fit.x,arm)
    u,v=subspace(L2@Q),subspace(L2@studentQ);cos=np.linalg.svd(u.T@v,compute_uv=False)
    targetunit=L2@Q;targetunit/=np.linalg.norm(targetunit,axis=0)
    studentunit=L2@studentQ;studentunit/=np.linalg.norm(studentunit,axis=0)
    similarity=np.abs(targetunit.T@studentunit);ii,jj=linear_sum_assignment(-similarity)
    transfers={};readouts={}
    for labelcount in ['population',32,128]:
     X,Y=(L@studentK,L@held) if labelcount=='population' else (Phi[:labelcount]@studentK,Phi[:labelcount]@held)
     readout=np.linalg.lstsq(X,Y,rcond=1e-12)[0];readouts[str(labelcount)]=readout
     transfers[str(labelcount)]=np.sum((L@(studentK@readout-held))**2,axis=0).tolist()
    name=f'{seed}_{init}_{arm}'
    np.savez_compressed(out/f'{name}.npz',initial=v0,final=fit.x,teacher_A=A,teacher_B=B,teacher_W=W,teacher_coefficients=truth,scales=scales,heldout_coefficients=held,adaptation_inputs=adapt,**{'readout_'+k:v for k,v in readouts.items()})
    record=dict(name=name,teacher_seed=seed,initial_seed=init,arm=arm,parameters=len(fit.x),supervised_outputs=len(scales),mse=float(fun(fit.x)@fun(fit.x)),nfev=fit.nfev,njev=fit.njev,status=int(fit.status),message=fit.message,optimality=float(fit.optimality),seconds=time.monotonic()-start,
     hidden_rank=subspace(L2@studentQ).shape[1],principal_cosines=cos.tolist(),component_cosines=similarity[ii,jj].tolist(),transfer_mse=transfers,checkpoint_sha256=digest(out/f'{name}.npz'))
    write(out/f'{name}.json',record);print(name,record['mse'],record['component_cosines'],flush=True)
 write(out/'completed.json',dict(status='complete',fits=24))
if __name__=='__main__':main()

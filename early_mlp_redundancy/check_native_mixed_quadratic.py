"""M101 random tensor identities, implementation gauges and informative limiting toys."""
import json
import numpy as np
from fit_pca_encoder import V,sha
from native_mixed_quadratic import prediction,change
from native_projected_quadratic import write,project,prediction as restricted

def main():
 rng=np.random.default_rng(10101);rows=[]
 for case in range(12):
  L=rng.normal(size=(7,5));R=rng.normal(size=(7,5));D=rng.normal(size=(3,7));mean=rng.normal(size=5);E=rng.normal(size=(5,2));B=rng.normal(size=(5,2));z,w=rng.normal(size=(2,5));T=np.einsum('on,ni,nj->oij',D,L,R);T=(T+T.swapaxes(1,2))/2;u=z-mean;h=u-project(u,E,B);explicit=np.einsum('oij,i,j->o',T,z,z)-np.einsum('oij,i,j->o',T,h,h);actual=prediction(L,R,D,mean,E,B,z);assert np.allclose(actual,explicit,rtol=1e-11,atol=1e-9)
  finite=prediction(L,R,D,mean,E,B,w)-actual;analytic=change(L,R,D,mean,E,B,z,w);assert np.allclose(finite,analytic,rtol=1e-11,atol=1e-9)
  for copies,coeff in [(4,np.ones(4)/4),(2,np.array([2.,-1.]))]:
   ll=np.repeat(L,copies,axis=0);rr=np.repeat(R,copies,axis=0);dd=np.repeat(D,copies,axis=1)*np.tile(coeff,7);assert np.allclose(prediction(ll,rr,dd,mean,E,B,z),actual,rtol=1e-11,atol=1e-9)
  assert np.allclose(prediction(L,R,D,mean,np.eye(5),np.eye(5),z),write(L,R,D,z),rtol=1e-11,atol=1e-9)
  rows.append(dict(case=case,tensor_absolute_error=float(np.linalg.norm(actual-explicit)),finite_change_absolute_error=float(np.linalg.norm(finite-analytic))))
 L=np.array([[1.,0.]]);R=np.array([[0.,1.]]);D=np.ones((1,1));E=B=np.array([[1.],[0.]]);mean=np.zeros(2);X=np.array([[1.,2.],[2.,3.],[-1.,4.]]);native=write(L,R,D,X);mixed=prediction(L,R,D,mean,E,B,X);old=restricted(L,R,D,mean,E,B,X);assert np.allclose(mixed,native) and np.all(old==0)
 # Keeping mixed terms is not monotonically better when omitted terms cancel them.
 L2=np.array([[1.,0.],[0.,1.]]);R2=np.array([[0.,1.],[0.,1.]]);D2=np.array([[1.,-1.]]);x=np.array([2.,2.]);assert float(write(L2,R2,D2,x)[0])==float(restricted(L2,R2,D2,mean,E,B,x)[0])==0;assert float(prediction(L2,R2,D2,mean,E,B,x)[0])==4
 out=dict(status='passed',source_sha256=sha(__file__),module_sha256=sha(V.parent/'native_mixed_quadratic.py'),rows=rows,pure_mixed_toy_exact=True,cancellation_counterexample=dict(native=0,restricted=0,mixed=4),scope='All12tensor/finitechange/gaugechecks andfullrankcontrol; purecrossinteractionrecoveredwithonecoordinate; moretermsnotguaranteedbetterduefunctionaltermcancellation. No nativechanneloroptimizationhistoryclaim.')
 f=V/'native_mixed_quadratic_toys.json';assert not f.exists();f.write_text(json.dumps(out,indent=2)+'\n');print('PASSED',len(rows))
if __name__=='__main__':main()

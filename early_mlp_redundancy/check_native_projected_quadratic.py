"""Independent explicit tensor tests of integrability and the anchor-only counterexample."""
import json
import numpy as np
from fit_pca_encoder import V,sha
from native_projected_quadratic import prediction,change,action

def main():
 rng=np.random.default_rng(10002);rows=[]
 for case in range(12):
  L=rng.normal(size=(7,5));R=rng.normal(size=(7,5));D=rng.normal(size=(3,7));mean=rng.normal(size=5);E=rng.normal(size=(5,2));B=rng.normal(size=(5,2));P=E@B.T;z,w=rng.normal(size=(2,5));T=np.einsum('on,ni,nj->oij',D,L,R);T=(T+T.swapaxes(1,2))/2;shift=z-mean
  explicit=np.einsum('oij,i,j->o',T,mean,mean)+2*np.einsum('oij,i,j->o',T,mean,shift)+np.einsum('oij,i,j->o',T,P.T@shift,P.T@shift)
  error=float(np.linalg.norm(prediction(L,R,D,mean,E,B,z)-explicit));assert np.allclose(prediction(L,R,D,mean,E,B,z),explicit,rtol=1e-12,atol=1e-10)
  delta=prediction(L,R,D,mean,E,B,w)-prediction(L,R,D,mean,E,B,z);assert np.allclose(delta,change(L,R,D,mean,E,B,z,w),rtol=1e-12,atol=1e-10)
  v=rng.normal(size=5);h=1e-5;fd=(prediction(L,R,D,mean,E,B,z+h*v)-prediction(L,R,D,mean,E,B,z-h*v))/(2*h);analytic=action(L,R,D,mean,v)+action(L,R,D,(z-mean)@P,v@P);rel=float(np.linalg.norm(fd-analytic)/max(np.linalg.norm(analytic),1e-300));assert rel<1e-8
  assert np.allclose(prediction(L,R,D,mean,np.eye(5),np.eye(5),z),np.einsum('oij,i,j->o',T,z,z),rtol=1e-12,atol=1e-10)
  rows.append(dict(case=case,explicit_tensor_absolute_error=error,finite_difference_relative_error=rel))
 # F=z0*z1; retaining only context coordinate z1 gives anchor-only Jhat(z)=[z1,0].
 # Cross partials 1 and 0 disagree: it is not a Jacobian field of a scalar potential.
 L=np.array([[1.,0.]]);R=np.array([[0.,1.]]);D=np.ones((1,1));P=np.diag([0.,1.]);u=np.array([0.,1.]);v=np.array([1.,0.]);gap=float((action(L,R,D,u@P,v)-action(L,R,D,v@P,u))[0]);assert gap==1.
 out=dict(status='passed',source_sha256=sha(__file__),module_sha256=sha(V.parent/'native_projected_quadratic.py'),checks=rows,anchor_only_reciprocity_gap=gap,scope='Explicit symmetric polynomial tensor, finitechanges, centralfinite differences, exactfullrank recovery; anchor-only contextprojection can be nonintegrable. No native modelclaim.')
 p=V/'native_projected_quadratic_toys.json';assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n');print('PASSED',len(rows),'maxFD',max(x['finite_difference_relative_error'] for x in rows))
if __name__=='__main__':main()

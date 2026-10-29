"""M100 independent explicit-Jacobian checks and known-structure recovery toys."""
import json
import numpy as np
from fit_pca_encoder import V,sha
from native_operator_metric import jacobian,operator_metric,fit_metric_pca,reconstruct

def main():
 rng=np.random.default_rng(10001);records=[]
 for case in range(12):
  d=5;m=7;o=3;L=rng.normal(size=(m,d));R=rng.normal(size=(m,d));D=rng.normal(size=(o,m));T=rng.normal(size=(d,3));S=T@T.T/3
  for name,cov in [('isotropic',np.eye(d)),('direction_weighted',S)]:
   Q=operator_metric(L,R,D,cov);Js=[jacobian(L,R,D,e) for e in np.eye(d)];explicit=np.array([[np.trace(a@cov@b.T) for b in Js] for a in Js]);error=float(np.linalg.norm(Q-explicit)/np.linalg.norm(explicit));assert error<1e-12
   a,b=rng.normal(size=(2,d));j=jacobian(L,R,D,a)-jacobian(L,R,D,b);want=float(np.trace(j@cov@j.T));got=float((a-b)@Q@(a-b));assert np.isclose(got,want,rtol=1e-12,atol=1e-12)
   # Split each native channel into four equal contributions; same operator metric.
   L4=np.repeat(L,4,axis=0);R4=np.repeat(R,4,axis=0);D4=np.repeat(D/4,4,axis=1);assert np.allclose(operator_metric(L4,R4,D4,cov),Q,rtol=1e-12,atol=1e-12)
   # Opposing 2 and -1 copies also preserve the function and metric.
   L2=np.repeat(L,2,axis=0);R2=np.repeat(R,2,axis=0);D2=np.repeat(D,2,axis=1)*np.tile([2.,-1.],m);assert np.allclose(operator_metric(L2,R2,D2,cov),Q,rtol=1e-12,atol=1e-12)
   records.append(dict(case=case,metric=name,relative_error=error))
 # F(z)=z0*z1, independent of high-variance z2. Fourfold balanced grid makes covariance exact.
 X=np.array([[a,b,c] for a in [-1.,1.] for b in [-2.,2.] for c in [-100.,100.]])
 L=np.array([[1.,0.,0.]]);R=np.array([[0.,1.,0.]]);D=np.ones((1,1));Q=operator_metric(L,R,D);metric=fit_metric_pca(X,Q,2);euclidean=fit_metric_pca(X,np.eye(3),2);qm=reconstruct(metric,X);qe=reconstruct(euclidean,X)
 def error(Z):return float(sum(np.linalg.norm(jacobian(L,R,D,x)-jacobian(L,R,D,z))**2 for x,z in zip(X,Z))/sum(np.linalg.norm(jacobian(L,R,D,x))**2 for x in X))
 me,ee=error(qm),error(qe);assert me<1e-24 and np.isclose(ee,.2,atol=1e-12)
 # Full metric rank must recover all observable operator variation, including singular Q.
 assert metric['metric_rank']==2
 # An observable channel may be unseen under a direction-only metric: explicitly preserve this limitation.
 S=np.diag([1.,0.,0.]);narrow=fit_metric_pca(X,operator_metric(L,R,D,S),1);narrow_error=error(reconstruct(narrow,X));assert np.isclose(narrow_error,.2,atol=1e-12)
 out=dict(status='passed',source_sha256=sha(__file__),module_sha256=sha(V.parent/'native_operator_metric.py'),random_checks=records,toy=dict(function='z0*z1 with irrelevant highvariance z2',metric_rank=2,operator_metric_rank2_error=me,Euclidean_rank2_error=ee,direction_only_rank1_full_operator_error=narrow_error),scope='Exact explicit-Jacobian metriccheck; knownshared lowrank recovery; duplicated andsignedcancelling implementations identicalmetric; directionalblindspot demonstrated. Doesnotestablishnative structure orsemanticunits.')
 f=V/'native_operator_metric_toys.json';assert not f.exists();f.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()

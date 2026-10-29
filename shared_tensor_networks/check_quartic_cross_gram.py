import itertools,json
from pathlib import Path
import numpy as np
from quartic_cross_gram import cross_gram,product_factors,sphere_radial_moment
from exact_scaling_task_span import feature_gram
rng=np.random.default_rng(235001);records=[]
for d in [2,3,4]:
 L=rng.normal(size=(7,4,d));R=rng.normal(size=(5,4,d))
 nodes,weights=np.polynomial.hermite.hermgauss(5);idx=np.array(list(itertools.product(range(5),repeat=d)))
 x=np.sqrt(2)*nodes[idx];w=np.prod(weights[idx]/np.sqrt(np.pi),axis=1)
 fl=np.prod(np.einsum('nd,pkd->npk',x,L),axis=2);fr=np.prod(np.einsum('nd,pkd->npk',x,R),axis=2)
 expected=fl.T@(w[:,None]*fr);actual=cross_gram(L,R,block=3)
 error=np.linalg.norm(expected-actual)/max(1.,np.linalg.norm(expected));assert error<1e-12
 np.testing.assert_allclose(actual,cross_gram(L,R,block=64),rtol=1e-12,atol=1e-12)
 np.testing.assert_allclose(actual,cross_gram(R,L,block=2).T,rtol=1e-12,atol=1e-12)
 A=rng.normal(size=(3,d));B=rng.normal(size=(3,d));scale=np.array([.7,1.,2.]);F=product_factors(A,B,scale)
 np.testing.assert_allclose(cross_gram(F,F),feature_gram(A,B,scale),rtol=1e-12,atol=1e-10)
 # E||X||^8 = d(d+2)(d+4)(d+6), independently via quadrature.
 radial=np.sum(w*np.sum(x*x,axis=1)**4);np.testing.assert_allclose(radial,sphere_radial_moment(d),rtol=1e-12)
 records.append(dict(d=d,quadrature_relative_error=error,radial_moment=float(radial)))
path=Path('results/quartic_cross_gram');path.mkdir(exist_ok=True);(path/'checks.json').write_text(json.dumps(records,indent=2)+'\n');print(records)

import itertools,json
from pathlib import Path
import numpy as np
import torch
from coverage_evaluation import features,empirical_mse,components,teacher_geometry,candidate_geometry,population_errors,adaptation
from multitask import monomials,product_map
from experiment import gram

records=[]
for d,h in [(4,2),(6,3)]:
 rng=np.random.default_rng(258001+d);p=h*(h+1)//2
 A,B,TA,TB=[rng.normal(size=(h,d))/np.sqrt(d) for _ in range(4)]
 W=rng.normal(size=(5,p));TW=rng.normal(size=(5,p));x=rng.normal(size=(33,d));noise=rng.normal(size=(33,5))*.05
 keys=monomials(d);M=gram(keys);F=product_map(A,B,keys);T=product_map(TA,TB,keys)
 monomial=np.prod(x[:,None,:]**np.array(keys)[None,:,:],axis=2);y=monomial@T@TW.T+noise
 np.testing.assert_allclose(features(A,B,x),monomial@F,atol=1e-10,rtol=1e-10)
 direct=np.mean((monomial@F@W.T-y)**2)
 for block in [1,7,256]:np.testing.assert_allclose(empirical_mse(A,B,W,x,y,block),direct,rtol=1e-12,atol=1e-12)
 teacher=teacher_geometry(TA,TB,TW);candidate=candidate_geometry(A,B,teacher)
 error=F@W.T-T@TW.T;expected=np.sum(error*(M@error),axis=0)
 actual=population_errors(W.T,candidate['G'],candidate['C'],TW,teacher['energy'])
 np.testing.assert_allclose(actual,expected,atol=1e-9,rtol=1e-10)
 for budget in [p-1,2*p]:
  coef,result=adaptation(A,B,x,y,candidate['G'],candidate['C'],TW,teacher['energy'],budget)
  X=monomial[:budget]@F;u,s,vh=np.linalg.svd(X,full_matrices=False);rank=s>s[0]*1e-12
  independent=vh[rank].T@((u[:,rank].T@y[:budget])/s[rank,None]);np.testing.assert_allclose(coef,independent,rtol=1e-9,atol=1e-9)
  error=F@coef-T@TW.T;np.testing.assert_allclose(result['errors'],np.sum(error*(M@error),axis=0),atol=1e-8,rtol=1e-9)
 np.testing.assert_allclose(components(A,B,A,B),1,atol=1e-12)
 np.testing.assert_array_equal(components(np.zeros_like(A),B,TA,TB),np.zeros(h))
 records.append(dict(d=d,h=h,population_max_error=float(np.max(np.abs(actual-expected))),budgets=[p-1,2*p]))
# Real saved h64 model: independent Torch float64 literal product/readout forward.
path=Path('/tmp/task_coverage_h64_smoke/256101/n8192_t520/257001.npz')
with np.load(path) as saved:A,B,W=[saved[k].astype(float) for k in ['A','B','W']]
x=np.random.default_rng(258101).normal(size=(32,128));z=(torch.tensor(x)@torch.tensor(A).T)*(torch.tensor(x)@torch.tensor(B).T)
phi=torch.stack([z[:,i]*z[:,j] for i,j in itertools.combinations_with_replacement(range(64),2)],dim=1)
expected=(phi@torch.tensor(W[:520]).T).numpy();actual=features(A,B,x)@W[:520].T
np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-10)
root=Path('results/coverage_evaluation');root.mkdir(exist_ok=True)
(root/'checks.json').write_text(json.dumps(dict(small_polynomial_fixtures=records,h64_forward_max_error=float(np.max(np.abs(actual-expected)))),indent=2)+'\n');print('Evaluation checks passed',records)

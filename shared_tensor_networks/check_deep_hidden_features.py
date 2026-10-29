import itertools,json
from pathlib import Path
import numpy as np
import torch
from deep_multitask import DeepMultiTask
from depth3_hidden_transfer import features,numpy_features
from quartic_cross_gram import product_factors,cross_gram
records=[]
for depth in [2,3]:
 for shared in [False,True]:
  net=DeepMultiTask(3,2,12,depth,shared,237001);state={k:v.detach().numpy() for k,v in net.state_dict().items()};x=np.random.default_rng(237002).normal(size=(11,3))
  with torch.no_grad():ft=features(net,torch.tensor(x)).numpy();pt=net(torch.tensor(x)).numpy()
  fn,pn=numpy_features(state,x);np.testing.assert_allclose(ft,fn,rtol=1e-10,atol=1e-10);np.testing.assert_allclose(pt,pn,rtol=1e-10,atol=1e-10)
  fq,pq=numpy_features(state,x*1.1);np.testing.assert_allclose(fq,fn*1.1**(2**depth),rtol=1e-10,atol=1e-10)
  if depth==2:
   F=product_factors(state['A.0'],state['B.0'],state['scale_0']) if shared else np.concatenate([product_factors(state['A.0'][i],state['B.0'][i],state['scale_0'][i]) for i in range(12)])
   nodes,w=np.polynomial.hermite.hermgauss(5);ind=np.array(list(itertools.product(range(5),repeat=3)));xx=np.sqrt(2)*nodes[ind];weights=np.prod(w[ind]/np.sqrt(np.pi),axis=1)
   phi,_=numpy_features(state,xx);expected=phi.T@(weights[:,None]*phi);actual=cross_gram(F,F);np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-10)
  records.append(dict(depth=depth,shared=shared,passed=True))
Path('results/depth3_hidden_transfer/feature_checks.json').write_text(json.dumps(records,indent=2)+'\n');print('Four independent feature/degree checks and two exact-moment fixtures passed.')

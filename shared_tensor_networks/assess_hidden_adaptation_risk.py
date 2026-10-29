"""Exact conditional bias/noise decomposition of fixed linear adaptation."""
import json
from pathlib import Path
import numpy as np
from experiment import gram
from multitask import monomials
from run_svhn_standard import digest,write
root=Path(__file__).parent;out=root/'results/hidden_extraction';audit=json.loads((out/'audit.json').read_text());assert audit['status']=='passed'
keys=monomials(8);G=gram(keys);L=np.linalg.cholesky(G).T;rows=[];worst=0.;worst_naive_trace=0.
for ref in audit['records']:
 path=out/(ref['name']+'.json');assert digest(path)==ref['result_sha256'];d=np.load(path.with_suffix('.npz'));assert digest(path.with_suffix('.npz'))==ref['checkpoint_sha256'];r=json.loads(path.read_text());data=np.load(root/r['data_file']);truth=data['truth'][:,10:];pa=np.prod(data['adaptation_inputs'][:,None,:]**np.array(keys)[None,:,:],axis=2)
 for extraction in ['outputs','hidden_union','quadratic_consensus','quartic_pca']:
  F=d['features_'+extraction]
  for count in [32,128]:
   X=pa[:count]@F;clean=pa[:count]@truth;inverse=np.linalg.pinv(X,rcond=1e-12);B0=inverse@clean;bias=F@B0-truth
   bias_mse=np.sum((L@bias)**2,axis=0);noise_variance=.05**2*float(np.linalg.norm(L@F@inverse)**2)
   naive=.05**2*float(np.trace((F.T@G@F)@(inverse@inverse.T)))
   worst_naive_trace=max(worst_naive_trace,abs(noise_variance-naive)/max(1,abs(noise_variance)))
   _,R=np.linalg.qr(L@F,mode='reduced');other=.05**2*float(np.linalg.norm(R@inverse)**2)
   error=abs(noise_variance-other)/max(1,abs(noise_variance));assert error<1e-7,(r['name'],extraction,count,error);worst=max(worst,error)
   actual=inverse@(clean+.05*data['adaptation_noise'][:count]);saved=d[f'readout_{extraction}_{count}'];np.testing.assert_allclose(F@actual,F@saved,atol=1e-7,rtol=1e-7)
   rows.append(dict(name=r['name'],method=r['method'],teacher_seed=r['teacher_seed'],extraction=extraction,labels=count,dimension=F.shape[1],conditional_bias_mse=bias_mse.tolist(),conditional_noise_variance=noise_variance,expected_mse=(bias_mse+noise_variance).tolist()))
write(out/'conditional_adaptation_risk.json',dict(status='complete',rows=rows,max_qr_formula_relative_error=worst,max_unstable_naive_trace_discrepancy=worst_naive_trace,source_sha256=digest(Path(__file__)),scope='exact expectation over fresh iid adaptation-label noise conditional on the fixed extraction,adaptation inputs,and teacher functions; no additional labels used in training or model selection'))
print('400conditional-risk QR/replay checks passed.',worst,'Naive trace discrepancy retained:',worst_naive_trace)

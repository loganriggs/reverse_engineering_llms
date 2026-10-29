"""Profile readout nuisance parameters; verify information and covariance ordering."""
from pathlib import Path
import numpy as np
from run_svhn_standard import write,digest
root=Path(__file__).parent;out=root/'results/readout_stability';records=[]
for seed in [218001,218002,218003]:
 infos=[];covs=[]
 for arm,m in [('one',1),('three',3),('six',6)]:
  d=np.load(out/f'{seed}_{arm}_geometry.npz');J=d['J'].reshape(126,m,-1)*np.sqrt(m);info=np.zeros((36,36))
  for t in range(m):
   hidden=J[:,t,:36];readout=J[:,t,36+6*t:36+6*(t+1)]
   residual=hidden-readout@np.linalg.lstsq(readout,hidden,rcond=1e-12)[0]
   info+=residual.T@residual
  values,vectors=np.linalg.eigh(info);assert np.sum(values>values[-1]*1e-10)==30
  inverse=(vectors[:,-30:]/values[-30:])@vectors[:,-30:].T
  F=d['F'][:,:36];cov=F@inverse@F.T/126
  expected=d['response']@d['response'].T/(126*m)
  error=float(np.linalg.norm(cov-expected)/max(1,np.linalg.norm(expected)));assert error<1e-7
  if infos:
   information_min=float(np.linalg.eigvalsh(info-infos[-1]).min());covariance_min=float(np.linalg.eigvalsh(covs[-1]-cov).min())
   assert information_min>-1e-8 and covariance_min>-1e-7
  else:information_min=covariance_min=None
  records.append(dict(seed=seed,arm=arm,profiled_rank=30,covariance_replay_error=error,min_added_information_eigenvalue=information_min,min_covariance_reduction_eigenvalue=covariance_min))
  infos.append(info);covs.append(cov)
write(out/'profiled_information_checks.json',dict(status='passed',records=records,source_sha256=digest(Path(__file__))))
print('Nine profiled information/covariance replays and six nested PSD comparisons passed.')

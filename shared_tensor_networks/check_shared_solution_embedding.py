"""A shared solution is a feasible independent-head empirical-loss certificate."""
import json
from pathlib import Path
import numpy as np
from direct_bilinear_readout import predict_jacobian
from run_svhn_standard import digest,write
root=Path(__file__).parent;out=root/'results/shared_separate_readouts';rows=[]
for seed in range(225001,225006):
 data=np.load(root/f'results/finite_readout_h4/{seed}_data.npz');x=data['inputs'][:512]
 tv=np.concatenate([data['teacher_A'].ravel(),data['teacher_B'].ravel(),data['teacher_W'][:10].ravel()])
 y=predict_jacobian(tv,x,4,False)/data['norms'][:10]+.05*data['noise'][:512,:10]
 for init in range(226001,226004):
  source=root/f'results/finite_readout_h4/{seed}_b5120_h10_noise0.05_random{init}.npz';v=np.load(source)['final'];direct=predict_jacobian(v,x,4,False)/data['norms'][:10]
  copied=np.stack([predict_jacobian(np.concatenate([v[:64],v[64:].reshape(10,10)[t]]),x,4,False)[:,0]/data['norms'][t] for t in range(10)],axis=1)
  np.testing.assert_allclose(copied,direct,atol=1e-11,rtol=1e-11)
  candidate=float(np.mean((copied-y)**2));shared=json.loads(source.with_suffix('.json').read_text());np.testing.assert_allclose(candidate,shared['train_mse'],atol=1e-10,rtol=1e-10)
  path=out/f'{seed}_{init}_separate_width.json';row=json.loads(path.read_text());actual=float(np.mean([h['train_mse'] for h in row['head_records']]))
  rows.append(dict(seed=seed,initialization=init,feasible_copied_shared_mse=candidate,independent_fitted_mse=actual,gap=actual-candidate,certified_suboptimal=actual>candidate+1e-9,independent_result_sha256=digest(path),shared_checkpoint_sha256=digest(source)))
write(out/'embedding_checks.json',dict(status='passed',rows=rows,certified_suboptimal=sum(r['certified_suboptimal'] for r in rows),source_sha256=digest(Path(__file__))))
print('Independent fits with explicit better feasible shared solution:',sum(r['certified_suboptimal'] for r in rows),'/15')

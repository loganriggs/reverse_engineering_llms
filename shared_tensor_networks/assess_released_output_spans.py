"""Teacher-informed output-span diagnostics; no training/selection uses this truth."""
import json
from pathlib import Path
import numpy as np
from experiment import gram
from multitask import monomials
from run_svhn_standard import digest,write
root=Path(__file__).parent;out=root/'results/released_readouts';L=np.linalg.cholesky(gram(monomials(8))).T
records=[]
for seed in range(225001,225006):
 data=np.load(root/f'results/finite_readout_h4/{seed}_data.npz');truth=data['truth'][:,:10];ut,st,_=np.linalg.svd(L@truth,full_matrices=False);assert (st>st[0]*1e-12).sum()==10
 for origin,methods in [('shared_separate_readouts',['shared','separate_width']),('released_readouts',['released','teacher'])]:
  for method in methods:
   for path in sorted((root/'results'/origin).glob(f'{seed}_*_{method}.json')):
    r=json.loads(path.read_text());d=np.load(path.with_suffix('.npz'));assert digest(path.with_suffix('.npz'))==r['checkpoint_sha256'];C=d['coefficients'];wc=L@C
    u,s,_=np.linalg.svd(wc,full_matrices=False);rank=int((s>s[0]*1e-12).sum());cos=np.linalg.svd(ut.T@u[:,:rank],compute_uv=False)
    off=wc-ut@(ut.T@wc);err=wc-L@truth
    held=L@data['truth'][:,10:];floor=float(np.mean(np.sum((held-u[:,:rank]@(u[:,:rank].T@held))**2,axis=0)))
    np.testing.assert_allclose(floor,np.mean(r['transfer_mse']['learned_population']),atol=1e-8,rtol=1e-8)
    bound=min(1.,float(np.linalg.norm(err,2)/st[-1]));actual_sine=float(np.sqrt(max(0.,1-cos.min()**2)));assert actual_sine<=bound+1e-9
    records.append(dict(name=r['name'],method=method,seed=seed,rank=rank,teacher_output_condition=float(st[0]/st[-1]),learned_output_condition=float(s[0]/s[-1]),minimum_span_cosine=float(cos.min()),span_sine=actual_sine,perturbation_bound=bound,off_teacher_span_mse=float(np.sum(off**2)/10),source_mse=float(np.sum(err**2)/10),transfer_floor=float(np.mean(r['transfer_mse']['learned_population'])),input_sha256=digest(path.with_suffix('.npz'))))
write(out/'span_diagnostics.json',dict(status='complete',records=records,source_sha256=digest(Path(__file__)),scope='teacher-informed descriptive projection/subspace analysis; not an intervention or training signal'))
for seed in range(225001,225006):
 rows=[r for r in records if r['seed']==seed and r['method'] in ['shared','teacher']]
 print(seed,[(r['method'],round(r['teacher_output_condition'],2),round(r['minimum_span_cosine'],4),r['off_teacher_span_mse'],r['transfer_floor']) for r in rows])

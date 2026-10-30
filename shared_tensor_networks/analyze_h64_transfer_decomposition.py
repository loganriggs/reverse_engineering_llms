"""Population refit decomposition of existing audited h64 transfer errors."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from regenerated_coverage import load_readout_chunks
from quartic_cross_gram import product_factors,cross_gram
from coverage_evaluation import population_errors
root=Path('results/task_coverage_h64');out=Path('results/h64_transfer_decomposition');out.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,value):
 p=out/name;tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)
rows=json.loads((root/'evaluation/results.json').read_text());audit=json.loads((root/'evaluation/audit.json').read_text());assert audit['status']=='passed' and audit['fits']==60
checked={r['name']:r for r in audit['records']}
for r in rows:assert checked[r['name']]['input_record_sha256']==hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest()
write('protocol.json',dict(scope='Post-hoc fixed-feature population refit; no scientific retraining',cuts=[1e-10,1e-12],budget=8192,sources={n:sha(n) for n in ['analyze_h64_transfer_decomposition.py','H64_TRANSFER_DECOMPOSITION_PLAN.md','quartic_cross_gram.py','coverage_evaluation.py']},results_sha256=sha(root/'evaluation/results.json'),audit_sha256=sha(root/'evaluation/audit.json')))
records=[];begin=time.perf_counter()
for seed in sorted({r['teacher'] for r in rows}):
 td=root/str(seed);meta=json.loads((td/'teacher.json').read_text());TW=load_readout_chunks(td,meta['readout_chunks'])[2080:]
 with np.load(td/'teacher_factors.npz') as z:TF=product_factors(z['A'],z['B'],np.ones(64))
 TG=cross_gram(TF,TF);energy=np.sum((TW@TG)*TW,axis=1);np.testing.assert_allclose(energy,1,atol=1e-9)
 for r in [r for r in rows if r['teacher']==seed]:
  print('Decomposing',r['name'],flush=True);path=td/f'n{r["n"]}_t{r["tasks"]}'/f'{r["initialization"]}.npz';assert sha(path)==r['checkpoint_sha256'] and sha(td/'teacher_factors.npz')==r['data_sha256']['teacher_factors.npz']
  with np.load(path) as z:F=product_factors(z['A'].astype(float),z['B'].astype(float),np.ones(64))
  G=cross_gram(F,F);C=cross_gram(F,TF);b=C@TW.T;scale=np.sqrt(np.diag(G));GN=G/scale[:,None]/scale[None,:];bn=b/scale[:,None]
  eig,V=eigh((GN+GN.T)/2,check_finite=False);assert min(eig)>-1e-8*max(eig)
  af=root/'evaluation'/f'{r["name"]}.npz';assert sha(af)==r['adaptation_sha256']
  with np.load(af) as z:u=z['learned_8192']*scale[:,None]
  actual=np.sum(u*(GN@u),axis=0)-2*np.sum(u*bn,axis=0)+energy
  np.testing.assert_allclose(actual,r['transfer']['learned_8192']['errors'],rtol=1e-7,atol=1e-8)
  cuts=[]
  for cutoff in [1e-10,1e-12]:
   keep=eig>max(eig)*cutoff;v=V[:,keep];star=v@((v.T@bn)/eig[keep,None]);floor=np.sum(star*(GN@star),axis=0)-2*np.sum(star*bn,axis=0)+energy
   delta=u-star;excess=np.sum(delta*(GN@delta),axis=0);cross=2*np.sum(delta*(GN@star-bn),axis=0)
   np.testing.assert_allclose(actual,floor+excess+cross,rtol=1e-7,atol=1e-8);assert min(floor)>-1e-7 and max(floor)<=max(energy)+1e-7 and min(excess)>-1e-7
   replay=None
   if cutoff==1e-12:
    ref,_,rank,_=np.linalg.lstsq(GN,bn,rcond=cutoff);assert rank==sum(keep)
    referror=np.sum(ref*(GN@ref),axis=0)-2*np.sum(ref*bn,axis=0)+energy;np.testing.assert_allclose(referror,floor,rtol=1e-6,atol=1e-8);replay=float(max(abs(referror-floor)))
   cuts.append(dict(cutoff=cutoff,rank=int(sum(keep)),population_refit_error=floor.tolist(),excess_readout_error=excess.tolist(),cross_term=cross.tolist(),independent_refit_error_difference=replay))
  records.append(dict(name=r['name'],teacher=seed,n=r['n'],tasks=r['tasks'],initialization=r['initialization'],finite_label_error=actual.tolist(),cuts=cuts,min_normalized_eigenvalue=float(min(eig)),max_normalized_eigenvalue=float(max(eig))))
  write('progress.json',dict(records=records,completed=len(records),seconds=time.perf_counter()-begin))
assert len(records)==60;write('results.json',dict(status='completed',records=records,seconds=time.perf_counter()-begin));print('Completed 60 decompositions',flush=True)

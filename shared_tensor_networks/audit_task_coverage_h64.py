"""Audit immutable completed training/evaluation; never launches training."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from regenerated_coverage import blocks,digest,load_readout_chunks
from coverage_audit import phi,factors,svd_refit,audit_case
from quartic_cross_gram import cross_gram

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def write(path,value):
 temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)

def training_check(root,smoke):
 protocol=json.loads((root/'protocol.json').read_text());assert protocol['smoke']==smoke and protocol['steps']==(2 if smoke else 3000)
 for name,digest_value in protocol['sources'].items():assert sha(Path(name))==digest_value
 complete=json.loads((root/'training_complete.json').read_text());expected=12 if smoke else 60
 assert complete['fits']==expected and complete['groups']==expected//3 and complete['status']=='trained_not_evaluated'
 progress=json.loads((root/'training_progress.json').read_text());groups=progress['groups'];assert len(groups)==expected//3 and progress['completed_fits']==expected
 teachers=list(range(256101,256102 if smoke else 256106));seen=set();records=[]
 initials={}
 for key in ['257001','257002']:
  with np.load(root/'initials'/f'{key}.npz') as z:initials[key]={k:z[k] for k in ['A','B','W']}
 for seed in teachers:
  directory=root/str(seed);meta=json.loads((directory/'teacher.json').read_text());TW=load_readout_chunks(directory,meta['readout_chunks']);assert array_sha(TW)==meta['readout_sha256']
  with np.load(directory/'teacher_factors.npz') as z:initials['teacher']=dict(A=z['A'].astype(np.float32),B=z['B'].astype(np.float32),W=TW[:2080].astype(np.float32))
  for group in [g for g in groups if g['teacher']==seed]:
   n,t=group['n'],group['tasks'];assert n in [8192,32768] and t in [520,2080];identity=(seed,n,t);assert identity not in seen;seen.add(identity)
   path=directory/f'n{n}_t{t}';assert json.loads((path/'trained.json').read_text())==group
   assert group['steps']==protocol['steps'] and group['active_parameters']==16384+t*2080 and group['status']=='trained'
   resource=group['resources'];assert resource['seconds']>0 and 0<resource['peak_allocated_bytes']<=resource['peak_reserved_bytes']
   curves=json.loads((path/'curves.json').read_text());expected_steps=[2] if smoke else list(range(200,3001,200));assert [c['step'] for c in curves]==expected_steps
   for c in curves:
    for key in ['preupdate_loss','gradient_norm']:assert len(c[key])==3 and np.isfinite(c[key]).all() and min(c[key])>=0
   assert [c['initialization'] for c in group['cases']]==['257001','257002','teacher']
   for case in group['cases']:
    assert case['file']==case['initialization']+'.npz'
    file=path/case['file'];assert sha(file)==case['sha256'] and file.stat().st_size<20*1024**2;assert 0<=case['clipped_steps']<=protocol['steps']
    with np.load(file) as z:
     for k,shape in [('A',(64,128)),('B',(64,128)),('W',(2080,2080))]:
      assert z[k].shape==shape and z[k].dtype==np.float32 and np.isfinite(z[k]).all()
      assert array_sha(initials[case['initialization']][k])==case['initial_sha256'][k]
     np.testing.assert_array_equal(z['W'][t:],initials[case['initialization']]['W'][t:])
    records.append(dict(teacher=seed,n=n,tasks=t,initialization=case['initialization'],checkpoint_sha256=case['sha256']))
 assert seen=={(seed,n,t) for seed in teachers for n in [8192,32768] for t in [520,2080]} and len(records)==expected
 return dict(status='training_structure_verified_not_metrics',fits=len(records),records=records)

def main(root,smoke=False,training_only=False):
 train=training_check(root,smoke)
 if training_only:write(root/'audit_training.json',train);print('Training structure checked',train['fits']);return
 out=root/'evaluation';rows=json.loads((out/'results.json').read_text());complete=json.loads((out/'complete.json').read_text());assert len(rows)==train['fits']==complete['fits'] and complete['status']=='evaluated_not_audited'
 assert len({r['name'] for r in rows})==len(rows)
 assert complete['transfer_metrics']==4*len(rows) and complete['learned_transfer_fits']==2*len(rows) and complete['oracle_transfer_fits']==2*len({r['teacher'] for r in rows})
 assert {(r['teacher'],r['n'],r['tasks'],r['initialization']) for r in rows}=={(r['teacher'],r['n'],r['tasks'],r['initialization']) for r in train['records']}
 for r in rows:
  assert r['name']==f"{r['teacher']}_n{r['n']}_t{r['tasks']}_{r['initialization']}" and r['active_parameters']==16384+r['tasks']*2080
 evaluation_protocol=json.loads((out/'protocol.json').read_text())
 for name,value in evaluation_protocol['source_sha256'].items():assert sha(Path(name))==value
 assert evaluation_protocol['training_protocol_sha256']==sha(root/'protocol.json')
 cache=out/'audit_cases';cache.mkdir(exist_ok=True);sources=['audit_task_coverage_h64.py','coverage_audit.py','regenerated_coverage.py','quartic_cross_gram.py']
 provenance=dict(source_sha256={n:sha(Path(n)) for n in sources},evaluation_protocol_sha256=sha(out/'protocol.json'),results_sha256=sha(out/'results.json'))
 if (cache/'protocol.json').exists():assert json.loads((cache/'protocol.json').read_text())==provenance
 else:write(cache/'protocol.json',provenance)
 checked=[]
 for seed in sorted({r['teacher'] for r in rows}):
  directory=root/str(seed);meta=json.loads((directory/'teacher.json').read_text());TW=load_readout_chunks(directory,meta['readout_chunks']);P=2080
  with np.load(directory/'teacher_factors.npz') as z:TA,TB=z['A'],z['B']
  TF=factors(TA,TB);TG=cross_gram(TF,TF);energy=np.sum((TW@TG)*TW,axis=1);np.testing.assert_allclose(energy,1,atol=1e-9)
  recipe=json.loads((directory/'source_recipe.json').read_text());assert (recipe['seed'],recipe['stream'],recipe['n'],recipe['tasks'],recipe['block_size'],recipe['noise_sd'])==(seed+5000,0,32768,2080,256,.05)
  x=np.empty((32768,128));y=np.empty((32768,P));assert len(recipe['blocks'])==128
  for b,record in zip(blocks(TA,TB,TW[:P],seed+5000,32768),recipe['blocks']):
   assert b['index']==record['index'] and record['rows']==256 and digest(b)==record['sha256']
   np.testing.assert_allclose(phi(TA,TB,b['x'])@TW[:P].T+.05*b['noise'],b['y'],rtol=1e-10,atol=1e-10)
   sl=slice(b['index']*256,(b['index']+1)*256);x[sl]=b['x'];y[sl]=b['y']
  with np.load(directory/'adaptation.npz') as z:xa,ya,noise=z['x'],z['y'],z['noise']
  for b in blocks(TA,TB,TW[P:],seed+5000,8192,stream=1):
   sl=slice(b['index']*256,(b['index']+1)*256)
   for key,array in [('x',xa),('y',ya),('noise',noise)]:np.testing.assert_array_equal(b[key],array[sl])
  np.testing.assert_allclose(phi(TA,TB,xa)@TW[P:].T+.05*noise,ya,rtol=1e-10,atol=1e-10)
  oracle=json.loads((out/f'{seed}_oracle.json').read_text())
  with np.load(out/f'{seed}_oracle.npz') as coeff:
   X=phi(TA,TB,xa)
   for k in [2048,8192]:
    coef=coeff[f'oracle_{k}'];assert svd_refit(X[:k],ya[:k],coef)==oracle[str(k)]['rank']
    diff=coef-TW[P:].T;err=np.sum(diff*(TG@diff),axis=0);np.testing.assert_allclose(err,oracle[str(k)]['errors'],rtol=1e-7,atol=1e-8)
  data_hashes={name:sha(directory/name) for name in ['teacher.json','teacher_factors.npz','source_recipe.json','adaptation.npz']}
  for r in [r for r in rows if r['teacher']==seed]:
   n,t=r['n'],r['tasks'];path=directory/f'n{n}_t{t}'/f'{r["initialization"]}.npz';array_path=out/f'{r["name"]}.npz'
   assert sha(path)==r['checkpoint_sha256'] and sha(array_path)==r['adaptation_sha256'] and r['data_sha256']==data_hashes
   for k in [2048,8192]:assert r['transfer'][f'oracle_{k}']==oracle[str(k)]
   record_sha=hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest();saved_path=cache/f'{r["name"]}.json'
   if saved_path.exists():
    checked_case=json.loads(saved_path.read_text());assert checked_case['input_record_sha256']==record_sha;checked.append(checked_case);continue
   print('Auditing',r['name'],flush=True)
   with np.load(path) as z:A,B,W=[z[k].astype(float) for k in ['A','B','W']]
   with np.load(array_path) as z:coeff={k:z[k] for k in z.files}
   result=audit_case(A,B,W,TA,TB,TW,x[:n],y[:n],xa,ya,coeff,r,(TF,TG,energy))
   checked_case=dict(name=r['name'],input_record_sha256=record_sha,**result);write(saved_path,checked_case);checked.append(checked_case)
 assert len(checked)==len(rows)
 write(out/'audit.json',dict(status='passed',fits=len(rows),learned_transfer_fits=2*len(rows),oracle_transfer_fits=2*len({r['teacher'] for r in rows}),training=train,records=checked,
  scope='Independent literal forwards, coefficient feature metric and explicit SVD; exact errors share the verified Wick kernel. Does not replay all 3000 optimizer steps or independently measure GPU memory.'))
 print('Audit passed',len(rows))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('results/task_coverage_h64'));parser.add_argument('--smoke',action='store_true');parser.add_argument('--training-only',action='store_true');args=parser.parse_args();main(args.root,args.smoke,args.training_only)

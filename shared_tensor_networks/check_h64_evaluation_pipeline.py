"""Small full-pipeline fixture: artifacts, resume, independent metric replay."""
import hashlib,json,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
from regenerated_coverage import blocks,digest,save_readout_chunks
from quartic_cross_gram import product_factors,cross_gram
from multitask import monomials,product_map
from experiment import gram
from coverage_audit import audit_case

def dump(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
rng=np.random.default_rng(259001);keys=monomials(4);M=gram(keys)
with tempfile.TemporaryDirectory(prefix='h64_evaluation_fixture_') as temp:
 root=Path(temp);directory=root/'1';directory.mkdir();dump(root/'protocol.json',dict(smoke=True,steps=2))
 TA,TB=[rng.normal(size=(2,4)) for _ in range(2)];TW=rng.normal(size=(7,3));TF=product_factors(TA,TB,np.ones(2));G=cross_gram(TF,TF)
 TW/=np.sqrt(np.sum((TW@G)*TW,axis=1))[:,None]
 np.savez_compressed(directory/'teacher_factors.npz',A=TA,B=TB)
 chunks=save_readout_chunks(directory,'teacher_W',TW)
 dump(directory/'teacher.json',dict(source_heads=3,readout_chunks=chunks,readout_sha256=hashlib.sha256(np.ascontiguousarray(TW).tobytes()).hexdigest()))
 source=list(blocks(TA,TB,TW[:3],259101,32));adapt=list(blocks(TA,TB,TW[3:],259101,8192,stream=1))
 dump(directory/'source_recipe.json',dict(seed=259101,n=32,stream=0,block_size=256,noise_sd=.05,blocks=[dict(index=b['index'],sha256=digest(b)) for b in source]))
 xa=np.concatenate([b['x'] for b in adapt]);ya=np.concatenate([b['y'] for b in adapt]);np.savez_compressed(directory/'adaptation.npz',x=xa,y=ya)
 groups=[]
 for n in [16,32]:
  for tasks in [1,3]:
   dest=directory/f'n{n}_t{tasks}';dest.mkdir();cases=[]
   for init in ['first','second','teacher']:
    A,B=(TA,TB) if init=='teacher' else (rng.normal(size=(2,4)),rng.normal(size=(2,4)))
    W=TW[:3] if init=='teacher' else rng.normal(size=(3,3));file=dest/f'{init}.npz'
    np.savez_compressed(file,A=A.astype(np.float32),B=B.astype(np.float32),W=W.astype(np.float32));cases.append(dict(file=file.name,initialization=init,sha256=sha(file)))
   group=dict(teacher=1,n=n,tasks=tasks,steps=2,active_parameters=16+3*tasks,cases=cases)
   dump(dest/'trained.json',group);groups.append(group)
 dump(root/'training_progress.json',dict(completed_fits=12,groups=groups));dump(root/'training_complete.json',dict(status='trained_not_evaluated',fits=12))
 command=[sys.executable,'evaluate_task_coverage_h64.py','--root',str(root),'--smoke']
 first=subprocess.run(command,capture_output=True,text=True);assert first.returncode==0,first.stdout+first.stderr
 out=root/'evaluation';original=(out/'results.json').read_bytes();records=json.loads(original);assert len(records)==12
 complete=json.loads((out/'complete.json').read_text());assert (complete['transfer_metrics'],complete['learned_transfer_fits'],complete['oracle_transfer_fits'])==(48,24,2)
 truth=product_map(TA,TB,keys)@TW.T
 for record in records:
  n,t=record['n'],record['tasks'];path=directory/f'n{n}_t{t}'/f'{record["initialization"]}.npz'
  with np.load(path) as data:A,B,W=[data[k].astype(float) for k in ['A','B','W']]
  F=product_map(A,B,keys);err=F@W[:t].T-truth[:,:t]
  np.testing.assert_allclose(record['population_mse'],np.sum(err*(M@err),axis=0),rtol=1e-9,atol=1e-8)
  with np.load(out/f'{record["name"]}.npz') as saved:
   for k in [2048,8192]:
    coef=saved[f'learned_{k}'];err=F@coef-truth[:,3:]
    np.testing.assert_allclose(record['transfer'][f'learned_{k}']['errors'],np.sum(err*(M@err),axis=0),rtol=1e-9,atol=1e-8)
   audit_case(A,B,W,TA,TB,TW,source[0]['x'][:n],source[0]['y'][:n],xa,ya,{key:saved[key] for key in saved.files},record)
   if record is records[0]:
    bad=json.loads(json.dumps(record));bad['population_mse'][0]+=max(1.,abs(bad['population_mse'][0]))*.1
    try:audit_case(A,B,W,TA,TB,TW,source[0]['x'][:n],source[0]['y'][:n],xa,ya,{key:saved[key] for key in saved.files},bad)
    except AssertionError:pass
    else:raise AssertionError('Auditor accepted a deliberately altered population error')
 second=subprocess.run(command,capture_output=True,text=True);assert second.returncode==0,second.stderr
 assert (out/'results.json').read_bytes()==original and 'Evaluating' not in second.stdout
 data_path=directory/'adaptation.npz';original_data=data_path.read_bytes();data_path.write_bytes(original_data+b'checksum-test')
 data_rejected=subprocess.run(command,capture_output=True,text=True);assert data_rejected.returncode!=0
 data_path.write_bytes(original_data)
 # Deliberate temp-fixture corruption must be rejected; no scientific file is touched.
 path=directory/'n16_t1'/'first.npz';path.write_bytes(path.read_bytes()+b'corruption-test')
 rejected=subprocess.run(command,capture_output=True,text=True);assert rejected.returncode!=0
result=dict(fits=12,learned_transfer_metrics=24,independent_case_audits=12,metric_corruption_rejected=True,first_exit=first.returncode,resume_exit=second.returncode,unchanged_resume=True,checkpoint_corruption_rejected=True,data_corruption_rejected=True,scope='Small synthetic full-pipeline fixture, not h64 scientific analysis')
destination=Path('results/coverage_evaluation');destination.mkdir(exist_ok=True);dump(destination/'pipeline_checks.json',result);print(result)

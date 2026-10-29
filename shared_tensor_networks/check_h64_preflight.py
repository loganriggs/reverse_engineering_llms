"""Audit saved two-step real-shape producer smoke outputs before full launch."""
import hashlib,json
from pathlib import Path
import numpy as np
from regenerated_coverage import blocks,digest,load_readout_chunks

root=Path('/tmp/task_coverage_h64_smoke');protocol=json.loads((root/'protocol.json').read_text());assert protocol['smoke'] and protocol['steps']==2
complete=json.loads((root/'training_complete.json').read_text());assert complete['groups']==4 and complete['fits']==12
teacher=root/'256101';meta=json.loads((teacher/'teacher.json').read_text());W=load_readout_chunks(teacher,meta['readout_chunks'])
assert hashlib.sha256(np.ascontiguousarray(W).tobytes()).hexdigest()==meta['readout_sha256']
with np.load(teacher/'teacher_factors.npz') as f:A,B=f['A'],f['B']
recipe=json.loads((teacher/'source_recipe.json').read_text());assert len(recipe['blocks'])==128
# Replay every source block and adaptation byte-for-byte; no source arrays are persisted.
for block,record in zip(blocks(A,B,W[:2080],261101,32768),recipe['blocks']):
 assert block['index']==record['index'] and digest(block)==record['sha256']
with np.load(teacher/'adaptation.npz') as data:
 for block in blocks(A,B,W[2080:],261101,8192,stream=1):
  sl=slice(block['index']*256,(block['index']+1)*256)
  for key in ['x','y','noise']:np.testing.assert_array_equal(block[key],data[key][sl])
initials={}
for seed in ['257001','257002']:
 with np.load(root/'initials'/f'{seed}.npz') as data:initials[seed]={k:data[k] for k in ['A','B','W']}
initials['teacher']=dict(A=A.astype(np.float32),B=B.astype(np.float32),W=W[:2080].astype(np.float32))
records=[]
for n in [8192,32768]:
 for tasks in [520,2080]:
  directory=teacher/f'n{n}_t{tasks}';record=json.loads((directory/'trained.json').read_text());assert record['steps']==2 and record['status']=='trained'
  for case in record['cases']:
   path=directory/case['file'];assert path.stat().st_size<20*1024**2
   assert hashlib.sha256(path.read_bytes()).hexdigest()==case['sha256']
   with np.load(path) as data:
    for key,shape in [('A',(64,128)),('B',(64,128)),('W',(2080,2080))]:
     value=data[key];assert value.shape==shape and value.dtype==np.float32 and np.isfinite(value).all()
     initial=initials[case['initialization']][key]
     assert hashlib.sha256(np.ascontiguousarray(initial).tobytes()).hexdigest()==case['initial_sha256'][key]
    np.testing.assert_array_equal(data['W'][tasks:],initials[case['initialization']]['W'][tasks:])
   records.append(dict(n=n,tasks=tasks,initialization=case['initialization'],sha256=case['sha256']))
out=Path('results/h64_preflight');out.mkdir(exist_ok=True)
(out/'checks.json').write_text(json.dumps(dict(fits=len(records),source_blocks=128,adaptation_blocks=32,records=records,producer_smoke=True,source_sha256=protocol['sources']),indent=2)+'\n')
print('Preflight passed:12real-shape checkpoints,128source blocks,32adaptation blocks; not scientific outcomes.')

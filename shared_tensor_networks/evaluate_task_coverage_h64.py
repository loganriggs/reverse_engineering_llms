"""Resume-safe post-training analysis; never starts or changes training."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from regenerated_coverage import blocks,digest,load_readout_chunks
from coverage_evaluation import teacher_geometry,candidate_geometry,population_errors,adaptation,empirical_mse,components

def filehash(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write_json(path,value):
 temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)
def main(root,smoke=False):
 protocol=json.loads((root/'protocol.json').read_text());assert protocol['smoke']==smoke
 complete=json.loads((root/'training_complete.json').read_text());expected=12 if smoke else 60
 assert complete['fits']==expected and complete['status']=='trained_not_evaluated'
 progress=json.loads((root/'training_progress.json').read_text());assert progress['completed_fits']==expected
 out=root/'evaluation';out.mkdir(exist_ok=True)
 sources=['evaluate_task_coverage_h64.py','coverage_evaluation.py','regenerated_coverage.py','quartic_cross_gram.py']
 provenance=dict(source_sha256={n:filehash(Path(n)) for n in sources},training_protocol_sha256=filehash(root/'protocol.json'),budgets=[2048,8192],numpy=np.__version__)
 if (out/'protocol.json').exists():assert json.loads((out/'protocol.json').read_text())==provenance,'Analysis source/protocol changed; do not mix runs.'
 else:write_json(out/'protocol.json',provenance)
 rows=[]
 for seed in sorted({g['teacher'] for g in progress['groups']}):
  directory=root/str(seed);meta=json.loads((directory/'teacher.json').read_text());TW=load_readout_chunks(directory,meta['readout_chunks']);P=meta['source_heads']
  data_hashes={name:filehash(directory/name) for name in ['teacher.json','teacher_factors.npz','source_recipe.json','adaptation.npz']}
  assert hashlib.sha256(np.ascontiguousarray(TW).tobytes()).hexdigest()==meta['readout_sha256']
  with np.load(directory/'teacher_factors.npz') as z:TA,TB=z['A'],z['B']
  teacher=teacher_geometry(TA,TB,TW);np.testing.assert_allclose(teacher['energy'],1,atol=1e-9)
  recipe=json.loads((directory/'source_recipe.json').read_text());x=np.empty((recipe['n'],TA.shape[1]));y=np.empty((recipe['n'],P))
  for count,block in enumerate(blocks(TA,TB,TW[:P],recipe['seed'],recipe['n'],stream=recipe['stream'],block_size=recipe['block_size'],noise_sd=recipe['noise_sd']),1):
   record=recipe['blocks'][count-1];assert block['index']==record['index'] and digest(block)==record['sha256']
   start=block['index']*recipe['block_size'];x[start:start+len(block['x'])]=block['x'];y[start:start+len(block['x'])]=block['y']
  assert count==len(recipe['blocks'])
  with np.load(directory/'adaptation.npz') as z:xa,ya=z['x'],z['y']
  oracle={};oracle_arrays={}
  for k in [2048,8192]:
   coef,record=adaptation(TA,TB,xa,ya,teacher['G'],teacher['G'],TW[P:],teacher['energy'][P:],k)
   oracle[str(k)]=record;oracle_arrays[f'oracle_{k}']=coef
  np.savez_compressed(out/f'{seed}_oracle.npz',**oracle_arrays);write_json(out/f'{seed}_oracle.json',oracle)
  for group in [g for g in progress['groups'] if g['teacher']==seed]:
   n,t=group['n'],group['tasks'];trained=directory/f'n{n}_t{t}'
   assert json.loads((trained/'trained.json').read_text())==group and group['steps']==protocol['steps']
   for case in group['cases']:
    checkpoint=trained/case['file'];assert filehash(checkpoint)==case['sha256']
    name=f'{seed}_n{n}_t{t}_{case["initialization"]}';record_path=out/f'{name}.json';array_path=out/f'{name}.npz'
    if record_path.exists():
     record=json.loads(record_path.read_text());assert record['checkpoint_sha256']==case['sha256'] and filehash(array_path)==record['adaptation_sha256'] and record['data_sha256']==data_hashes;rows.append(record);continue
    print('Evaluating',name,flush=True)
    with np.load(checkpoint) as z:A,B,W=[z[key].astype(float) for key in ['A','B','W']]
    candidate=candidate_geometry(A,B,teacher)
    source=population_errors(W[:t].T,candidate['G'],candidate['C'],TW[:t],teacher['energy'][:t]);assert min(source)>-1e-7 and np.isfinite(source).all()
    saved={};transfer={}
    for k in [2048,8192]:
     coef,result=adaptation(A,B,xa,ya,candidate['G'],candidate['C'],TW[P:],teacher['energy'][P:],k)
     saved[f'learned_{k}']=coef;transfer[f'learned_{k}']=result;transfer[f'oracle_{k}']=oracle[str(k)]
     assert np.isfinite(result['errors']).all()
    np.savez_compressed(array_path,**saved)
    record=dict(name=name,teacher=seed,n=n,tasks=t,initialization=case['initialization'],active_parameters=group['active_parameters'],
       train_mse=empirical_mse(A,B,W[:t],x[:n],y[:n,:t]),population_mse=source.tolist(),component_cosines=components(A,B,TA,TB).tolist(),
       transfer=transfer,checkpoint_sha256=case['sha256'],adaptation_sha256=filehash(array_path),data_sha256=data_hashes,source_recipe_sha256=filehash(directory/'source_recipe.json'))
    write_json(record_path,record);rows.append(record);write_json(out/'results.json',rows)
  del x,y,teacher,TW,TA,TB
 assert len(rows)==expected
 write_json(out/'results.json',rows);write_json(out/'complete.json',dict(status='evaluated_not_audited',fits=len(rows),transfer_fits=4*len(rows)))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('results/task_coverage_h64'));parser.add_argument('--smoke',action='store_true');args=parser.parse_args();main(args.root,args.smoke)

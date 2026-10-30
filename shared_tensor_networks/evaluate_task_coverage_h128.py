"""Resumable post-training scientific evaluation for chunked checkpoints."""
import argparse,fcntl,json
from pathlib import Path
import numpy as np
from h128_analysis_io import sha,write,training,data,case_path,case_name
from chunked_checkpoint import load_checkpoint
from quartic_cross_gram import product_factors,cross_gram
from coverage_evaluation import candidate_geometry,population_errors,adaptation,empirical_mse,components

def main(root):
 protocol,rows=training(root);out=root/'evaluation';out.mkdir(exist_ok=True);lock=(out/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 budgets=[protocol['adaptation']//4,protocol['adaptation']]
 sources=['evaluate_task_coverage_h128.py','h128_analysis_io.py','coverage_evaluation.py','chunked_checkpoint.py','quartic_cross_gram.py','regenerated_coverage.py']
 provenance=dict(training_protocol_sha256=sha(root/'protocol.json'),training_progress_sha256=sha(root/'progress.json'),budgets=budgets,numpy=np.__version__,sources={n:sha(Path(n)) for n in sources})
 if (out/'protocol.json').exists():assert json.loads((out/'protocol.json').read_text())==provenance
 else:write(out/'protocol.json',provenance)
 results=[]
 for seed in protocol['teachers']:
  TA,TB,TW,x,y,xa,ya,hashes=data(root,seed);p=protocol['p'];F=product_factors(TA,TB,np.ones(protocol['h']));G=cross_gram(F,F)
  energy=np.concatenate([np.sum((TW[i:i+128]@G)*TW[i:i+128],axis=1) for i in range(0,len(TW),128)]);np.testing.assert_allclose(energy,1,atol=1e-9)
  teacher=dict(F=F,G=G,energy=energy);oraclefile=out/f'{seed}_oracle.json';oraclearrays=out/f'{seed}_oracle.npz'
  if oraclefile.exists():
   oracle=json.loads(oraclefile.read_text());assert oracle['data_sha256']==hashes and oracle['arrays_sha256']==sha(oraclearrays)
  else:
   arrays={};oracle=dict(data_sha256=hashes,transfer={})
   for k in budgets:
    arrays[str(k)],oracle['transfer'][str(k)]=adaptation(TA,TB,xa,ya,G,G,TW[p:],energy[p:],k)
   np.savez(oraclearrays,**arrays);oracle['arrays_sha256']=sha(oraclearrays);write(oraclefile,oracle)
  for r in [r for r in rows if r['teacher']==seed]:
   name=case_name(r);case=case_path(root,r);recordfile=out/f'{name}.json';arrayfile=out/f'{name}.npz'
   # Hash-check every checkpoint chunk even when reusing evaluation records.
   A,B,W=[v.astype(float) for v in load_checkpoint(case/'checkpoint')]
   if recordfile.exists():
    record=json.loads(recordfile.read_text());assert record['training_record']==r and record['data_sha256']==hashes and record['arrays_sha256']==sha(arrayfile);results.append(record);continue
   print('Evaluating',name,flush=True);cg=candidate_geometry(A,B,teacher);t=r['tasks'];n=r['n']
   source=np.concatenate([population_errors(W[i:min(i+128,t)].T,cg['G'],cg['C'],TW[i:min(i+128,t)],energy[i:min(i+128,t)]) for i in range(0,t,128)])
   assert np.isfinite(source).all() and min(source)>-1e-7
   transfer={};saved={}
   for k in budgets:
    saved[str(k)],transfer[str(k)]=adaptation(A,B,xa,ya,cg['G'],cg['C'],TW[p:],energy[p:],k)
    assert np.isfinite(transfer[str(k)]['errors']).all() and min(transfer[str(k)]['errors'])>-1e-7
   np.savez(arrayfile,**saved)
   record=dict(name=name,training_record=r,data_sha256=hashes,arrays_sha256=sha(arrayfile),source_mse=source.tolist(),train_mse=empirical_mse(A,B,W[:t],x[:n],y[:n,:t]),component_cosines=components(A,B,TA,TB).tolist(),transfer=transfer,oracle_sha256=sha(oraclefile))
   write(recordfile,record);results.append(record);write(out/'results.json',results)
  del x,y,G,teacher,TW,TA,TB
 assert len(results)==len(rows);write(out/'results.json',results);write(out/'complete.json',dict(status='evaluated_not_audited',fits=len(rows),learned_transfer_fits=2*len(rows),oracle_transfer_fits=2*len(protocol['teachers'])))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);main(parser.parse_args().root)

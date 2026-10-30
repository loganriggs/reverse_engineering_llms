"""Independent metric and artifact audit; shared Wick kernel is explicitly scoped."""
import argparse,fcntl,json
from pathlib import Path
import numpy as np
from h128_analysis_io import sha,write,training,data,case_path,case_name
from chunked_checkpoint import load_checkpoint
from coverage_audit import phi,factors,matching,svd_refit
from quartic_cross_gram import cross_gram

def main(root):
 protocol,rows=training(root);out=root/'evaluation';assert json.loads((out/'complete.json').read_text())['fits']==len(rows)
 ep=json.loads((out/'protocol.json').read_text());assert ep['training_protocol_sha256']==sha(root/'protocol.json') and ep['training_progress_sha256']==sha(root/'progress.json')
 for n,h in ep['sources'].items():assert sha(Path(n))==h
 results=json.loads((out/'results.json').read_text());byname={r['name']:r for r in results};assert len(byname)==len(rows)==len(results) and set(byname)=={case_name(r) for r in rows}
 cache=out/'audit_cases';cache.mkdir(exist_ok=True);lock=(cache/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 provenance=dict(results_sha256=sha(out/'results.json'),evaluation_protocol_sha256=sha(out/'protocol.json'),sources={n:sha(Path(n)) for n in ['audit_task_coverage_h128.py','h128_analysis_io.py','coverage_audit.py','quartic_cross_gram.py','chunked_checkpoint.py']})
 if (cache/'protocol.json').exists():assert json.loads((cache/'protocol.json').read_text())==provenance
 else:write(cache/'protocol.json',provenance)
 checked=[];budgets=ep['budgets'];p=protocol['p']
 for seed in protocol['teachers']:
  TA,TB,TW,x,y,xa,ya,hashes=data(root,seed);td=root/str(seed);recipe=json.loads((td/'source_recipe.json').read_text())
  # Independently generate full-noise blocks and use literal product enumeration.
  for i,start in enumerate(range(0,len(x),256)):
   xx=np.random.default_rng(np.random.SeedSequence([recipe['seed'],0,i,0])).normal(size=(256,protocol['d']))
   noise=np.random.default_rng(np.random.SeedSequence([recipe['seed'],0,i,1])).normal(size=(256,p));count=min(256,len(x)-start)
   np.testing.assert_array_equal(x[start:start+count],xx[:count].astype(np.float32))
   yy=(phi(TA,TB,xx[:count])@TW[:p].T+.05*noise[:count]).astype(np.float32)
   np.testing.assert_allclose(y[start:start+count],yy,rtol=1e-6,atol=1e-6)
  TF=factors(TA,TB);TG=cross_gram(TF,TF);energy=np.concatenate([np.einsum('tp,pq,tq->t',TW[i:i+128],TG,TW[i:i+128],optimize=True) for i in range(0,len(TW),128)]);np.testing.assert_allclose(energy,1,atol=1e-9)
  oraclefile=out/f'{seed}_oracle.json';oracle=json.loads(oraclefile.read_text());assert oracle['data_sha256']==hashes and oracle['arrays_sha256']==sha(out/f'{seed}_oracle.npz')
  oraclecache=cache/f'{seed}_oracle.json'
  if oraclecache.exists():assert json.loads(oraclecache.read_text())['oracle_sha256']==sha(oraclefile)
  else:
   X=phi(TA,TB,xa)
   with np.load(out/f'{seed}_oracle.npz') as saved:
    for k in budgets:
     coef=saved[str(k)];rank=svd_refit(X[:k],ya[:k],coef);assert rank==oracle['transfer'][str(k)]['rank'];delta=coef-TW[p:].T
     error=np.einsum('pt,pq,qt->t',delta,TG,delta,optimize=True);np.testing.assert_allclose(error,oracle['transfer'][str(k)]['errors'],rtol=1e-7,atol=1e-8)
   del X;write(oraclecache,dict(oracle_sha256=sha(oraclefile)))
  for r in [r for r in rows if r['teacher']==seed]:
   name=case_name(r);case=case_path(root,r);record=byname[name];recordfile=out/f'{name}.json';arrayfile=out/f'{name}.npz';cached=cache/f'{name}.json'
   assert json.loads(recordfile.read_text())==record and record['training_record']==r and record['data_sha256']==hashes and record['arrays_sha256']==sha(arrayfile) and record['oracle_sha256']==sha(oraclefile)
   A,B,W=load_checkpoint(case/'checkpoint');t=r['tasks'];n=r['n']
   initial=TW[:p].astype(np.float32) if r['initialization']=='teacher' else load_checkpoint(root/'initials'/r['initialization'])[2]
   np.testing.assert_array_equal(W[t:],initial[t:]);del initial
   trained=json.loads((case/'trained.json').read_text());assert trained['status']=='trained_not_evaluated' and trained['active_parameters']==2*A.size+t*p and np.isfinite(trained['resources']['seconds']) and trained['resources']['seconds']>0
   curves=json.loads((case/'curves.json').read_text());assert [c['step'] for c in curves]==sorted(set(list(range(200,protocol['steps']+1,200))+[protocol['steps']]))
   assert all(np.isfinite(c['preupdate_loss']) and np.isfinite(c['gradient_norm']) for c in curves)
   if cached.exists():
    check=json.loads(cached.read_text());assert check['record_sha256']==sha(recordfile);checked.append(check);continue
   print('Auditing',name,flush=True);A,B,W=[v.astype(float) for v in [A,B,W]];F=factors(A,B);G=cross_gram(F,F);C=cross_gram(F,TF)
   source=[]
   for start in range(0,t,128):
    v=W[start:min(start+128,t)];tw=TW[start:min(start+128,t)]
    source.extend((np.einsum('tp,pq,tq->t',v,G,v,optimize=True)-2*np.sum((v@C)*tw,axis=1)+energy[start:min(start+128,t)]).tolist())
   np.testing.assert_allclose(source,record['source_mse'],rtol=1e-7,atol=1e-8)
   total=0.
   for start in range(0,n,127):
    end=min(start+127,n);delta=phi(A,B,x[start:end].astype(float))@W[:t].T-y[start:end,:t];total+=np.sum(delta**2)
   np.testing.assert_allclose(total/(n*t),record['train_mse'],rtol=1e-8,atol=1e-10)
   np.testing.assert_allclose(matching(A,B,TA,TB),record['component_cosines'],rtol=1e-8,atol=1e-10)
   X=phi(A,B,xa);ranks={}
   with np.load(arrayfile) as saved:
    for k in budgets:
     coef=saved[str(k)];rank=svd_refit(X[:k],ya[:k],coef);assert rank==record['transfer'][str(k)]['rank'];ranks[str(k)]=rank
     error=np.einsum('pt,pq,qt->t',coef,G,coef,optimize=True)-2*np.sum((coef.T@C)*TW[p:],axis=1)+energy[p:]
     np.testing.assert_allclose(error,record['transfer'][str(k)]['errors'],rtol=1e-7,atol=1e-8)
   del X
   check=dict(name=name,record_sha256=sha(recordfile),transfer_ranks=ranks,train_mse=float(total/(n*t)));write(cached,check);checked.append(check)
  del x,y,TW,TG
 write(out/'audit.json',dict(status='passed',fits=len(checked),records=checked,provenance_sha256=sha(cache/'protocol.json'),scope='All artifact/condition checks and independent metric plumbing; shared verified Wick kernel; no replay of every optimizer step or independent hardware-memory measurement'))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);main(parser.parse_args().root)

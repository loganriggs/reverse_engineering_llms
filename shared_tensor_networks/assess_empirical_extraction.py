"""Estimate extraction geometry from activations; use population moments only for evaluation."""
import json
from pathlib import Path
import numpy as np
from assess_hidden_extraction import products,design,keys,qkeys,L
from multitask import product_map
from direct_bilinear_readout import quadratic_coefficients
from run_svhn_standard import digest,write

def main():
 root=Path(__file__).parent;out=root/'results/empirical_extraction';out.mkdir(exist_ok=False)
 write(out/'protocol.json',dict(models=50,representations=300,pools=['source512','fresh512','fresh4096'],ranks=dict(quadratic_consensus=4,quartic_pca=10),source_sha256=digest(Path(__file__)),dependencies={s:digest(root/s) for s in ['assess_hidden_extraction.py','multitask.py','direct_bilinear_readout.py','experiment.py']}));(out/'source.py').write_bytes(Path(__file__).read_bytes())
 cache={}
 for seed in range(225001,225006):
  datafile=root/f'results/finite_readout_h4/{seed}_data.npz';data=np.load(datafile);fresh=np.random.default_rng(seed+7000).normal(size=(4096,8));np.savez_compressed(out/f'{seed}_unlabeled.npz',inputs=fresh)
  pools={'source512':data['inputs'][:512],'fresh512':fresh[:512],'fresh4096':fresh}
  cache[seed]=(data,{name:(design(x,qkeys),design(x,keys)) for name,x in pools.items()})
 for path in sorted((root/'results/hidden_extraction').glob('225*.json')):
  old=json.loads(path.read_text());source=root/old['input_file'];parent=json.loads(source.with_suffix('.json').read_text());src=np.load(source);assert digest(source)==old['input_sha256'];seed=old['teacher_seed'];data,pools=cache[seed]
  states=[np.load(root/parent['input_file'])['final']] if old['method']=='shared' else [src[f'final_{t}'] for t in range(10)]
  Q=[];K=[]
  for v in states:
   A=v[:32].reshape(4,8);B=v[32:64].reshape(4,8);Q.append(quadratic_coefficients(A,B));K.append(product_map(A,B,keys))
  Q=np.concatenate(Q,axis=1);K=np.concatenate(K,axis=1)
  for pool,(pq,pk) in pools.items():
   for extraction,raw,phi,rank in [('quadratic_consensus',Q,pq,4),('quartic_pca',K,pk,10)]:
    norms=np.sqrt(np.mean((phi@raw)**2,axis=0));normalized=raw/norms;Z=phi@normalized/np.sqrt(len(phi));_,s,vh=np.linalg.svd(Z,full_matrices=False);weights=vh[:rank].T/s[:rank];basis=normalized@weights
    F=products(basis) if extraction=='quadratic_consensus' else basis
    output_norm=np.sqrt(np.mean((pk@F)**2,axis=0));F/=output_norm
    # No teacher functions or adaptation labels enter extraction above.
    truth=data['truth'][:,10:];pa=design(data['adaptation_inputs'],keys);metrics=[];evidence={}
    for count in ['population',32,128]:
     X,Y=(L@F,L@truth) if count=='population' else (pa[:count]@F,pa[:count]@truth+.05*data['adaptation_noise'][:count])
     read=np.linalg.lstsq(X,Y,rcond=1e-12)[0];delta=F@read-truth;evidence['readout_'+str(count)]=read;metrics.append(dict(labels=count,mse=np.sum((L@delta)**2,axis=0).tolist()))
    name=f"{old['name']}_{pool}_{extraction}"
    np.savez_compressed(out/f'{name}.npz',features=F,basis=basis,raw_column_norms=norms,basis_weights=weights,singular_values=s,output_norms=output_norm,**evidence)
    write(out/f'{name}.json',dict(name=name,model_name=old['name'],method=old['method'],teacher_seed=seed,initial_seed=old['initial_seed'],pool=pool,extraction=extraction,rank=rank,unlabeled_inputs=len(phi),metrics=metrics,input_file=str(source.relative_to(root)),input_sha256=digest(source),data_file=old['data_file'],data_sha256=old['data_sha256'],checkpoint_sha256=digest(out/f'{name}.npz')))
  print(old['name'],'six extractions complete',flush=True)
 write(out/'completed.json',dict(status='complete',representations=300,models=50))
if __name__=='__main__':main()

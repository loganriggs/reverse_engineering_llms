"""Fixed unsupervised function-space compression of stored hidden dictionaries."""
import itertools,json
from pathlib import Path
import numpy as np
from experiment import gram,multiply
from multitask import monomials,product_map
from direct_bilinear_readout import quadratic_coefficients
from run_svhn_standard import digest,write
keys=monomials(8);L=np.linalg.cholesky(gram(keys)).T
qkeys=[tuple(pair.count(i) for i in range(8)) for pair in itertools.combinations_with_replacement(range(8),2)];L2=np.linalg.cholesky(gram(qkeys)).T

def products(Q):
 columns=[]
 for i,j in itertools.combinations_with_replacement(range(Q.shape[1]),2):
  poly=multiply(dict(zip(qkeys,Q[:,i])),dict(zip(qkeys,Q[:,j])));columns.append([poly.get(k,0.) for k in keys])
 return np.array(columns).T

def design(x,exponents):return np.prod(x[:,None,:]**np.array(exponents)[None,:,:],axis=2)
def normalized(C,root):return C/np.linalg.norm(root@C,axis=0)

def main():
 root=Path(__file__).parent;out=root/'results/hidden_extraction';out.mkdir(exist_ok=False)
 write(out/'protocol.json',dict(models=50,quadratic_rank=4,quartic_rank=10,labels=[32,128],source_sha256=digest(Path(__file__)),dependencies={s:digest(root/s) for s in ['experiment.py','multitask.py','direct_bilinear_readout.py']}));(out/'source.py').write_bytes(Path(__file__).read_bytes())
 for origin,methods in [('shared_separate_readouts',['shared','separate_width']),('released_readouts',['released','teacher'])]:
  for method in methods:
   for path in sorted((root/'results'/origin).glob(f'225*_{method}.json')):
    r=json.loads(path.read_text());checkpoint=path.with_suffix('.npz');d=np.load(checkpoint);assert digest(checkpoint)==r['checkpoint_sha256']
    datafile=root/r['data_file'];data=np.load(datafile);Q=[];union=[]
    states=[np.load(root/r['input_file'])['final']] if method=='shared' else [d[f'final_{t}'] for t in range(10)]
    for v in states:
     A=v[:32].reshape(4,8);B=v[32:64].reshape(4,8);Q.append(quadratic_coefficients(A,B));union.append(product_map(A,B,keys))
    Q=normalized(np.concatenate(Q,axis=1),L2);union=normalized(np.concatenate(union,axis=1),L)
    uq,sq,_=np.linalg.svd(L2@Q,full_matrices=False);compressed=np.linalg.solve(L2,uq[:,:4])
    uk,sk,_=np.linalg.svd(L@union,full_matrices=False)
    choices=dict(outputs=d['features'],hidden_union=union,quadratic_consensus=normalized(products(compressed),L),quartic_pca=np.linalg.solve(L,uk[:,:10]))
    # Extraction is complete before reading held-out labels/teacher alignment.
    truth=data['truth'];pa=design(data['adaptation_inputs'],keys);tq=L2@quadratic_coefficients(data['teacher_A'],data['teacher_B']);qt=np.linalg.qr(tq)[0]
    cos=np.linalg.svd(qt.T@uq[:,:4],compute_uv=False);evidence=dict(normalized_quadratics=Q,compressed_quadratics=compressed,quadratic_singular_values=sq,quartic_singular_values=sk)
    metrics=[]
    for extraction,F in choices.items():
     evidence['features_'+extraction]=F
     for count in ['population',32,128]:
      X,Y=(L@F,L@truth[:,10:]) if count=='population' else (pa[:count]@F,pa[:count]@truth[:,10:]+.05*data['adaptation_noise'][:count])
      read=np.linalg.lstsq(X,Y,rcond=1e-12)[0];diff=F@read-truth[:,10:];evidence[f'readout_{extraction}_{count}']=read
      metrics.append(dict(extraction=extraction,labels=count,dimension=F.shape[1],mse=np.sum((L@diff)**2,axis=0).tolist()))
    name=r['name'];np.savez_compressed(out/f'{name}.npz',**evidence)
    write(out/f'{name}.json',dict(name=name,method=method,teacher_seed=r['teacher_seed'],initial_seed=r['initial_seed'],metrics=metrics,consensus_principal_cosines=cos.tolist(),input_file=str(checkpoint.relative_to(root)),input_sha256=digest(checkpoint),data_file=r['data_file'],data_sha256=digest(datafile),checkpoint_sha256=digest(out/f'{name}.npz')))
    print(name,'minimum hidden-span cosine',float(cos.min()),flush=True)
 write(out/'completed.json',dict(status='complete',models=50))
if __name__=='__main__':main()

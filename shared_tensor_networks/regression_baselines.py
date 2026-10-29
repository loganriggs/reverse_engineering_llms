"""Fixed-feature least squares on exactly the E1 CPU datasets, with no test selection."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from experiment import gram, multiply, write_csv
from audit_results import forward


def feature_maps(weights, d, keys):
    unit = [tuple(int(i==j) for i in range(d)) for j in range(d)]
    def hidden(a,b):
        return [multiply(dict(zip(unit, aa)), dict(zip(unit, bb))) for aa,bb in zip(a,b)]
    left = hidden(weights['A'],weights['B'])
    right = hidden(weights['AR'],weights['BR']) if 'AR' in weights else left
    pairs = [(i,j) for i in range(len(left)) for j in range(len(right)) if 'AR' in weights or i<=j]
    hp = [multiply(left[i],right[j]) for i,j in pairs]
    def combine(w, polys):
        return {k:sum(wi*p.get(k,0.) for wi,p in zip(w,polys)) for k in set().union(*polys)}
    top = [multiply(combine(c,left),combine(e,right)) for c,e in zip(weights['C'],weights['D'])]
    return {name:np.array([[p.get(k,0.) for p in polys] for k in keys]) for name,polys in [('oracle_hidden',hp),('oracle_top',top)]}


def design(x,keys):
    return np.prod(x[:,None,:]**np.asarray(keys)[None,:,:],axis=2)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('source',type=Path);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();root=args.source;out=args.out;out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((root/'protocol.json').read_text())['arguments']
    assert cfg['device']=='cpu'
    torch.set_num_threads(1)
    grid=[0.,1e-10,1e-8,1e-6,1e-4,1e-3,1e-2,.1,1.,10.]
    protocol=dict(source_run=str(root),source_protocol_sha256=hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),ridge_grid=grid,
                  objective='mean squared loss + lambda * squared coefficient norm in each specified feature basis',
                  selection='minimum validation MSE; grid order breaks ties; test and population error never select',
                  solver='thin SVD; minimum norm cutoff 1e-12 times largest singular value',
                  oracle_access='teacher hidden features or teacher top products; all these features are privileged',
                  data='regenerate exact nested inputs from original frozen CPU generator; verify direct teacher forward')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    sizes=[int(v) for v in cfg['ntrain'].split(',')];rows=[];evidence={};checks=[]
    for kind in cfg['teacher'].split(','):
        for seed in range(cfg['seeds']):
            path=root/f'{kind}_{seed}_{sizes[0]}_tied.npz'
            with np.load(path) as data: weights={k.removeprefix('teacher_'):v.copy() for k,v in data.items() if k.startswith('teacher_')}
            # Enumerate all quartic monomials independent of producer expansion.
            import itertools
            keys=sorted({tuple(sum(i==j for i in inds) for j in range(cfg['d'])) for inds in itertools.combinations_with_replacement(range(cfg['d']),4)})
            H=gram(keys);L=np.linalg.cholesky(H)
            maps=feature_maps(weights,cfg['d'],keys)
            target=maps['oracle_top']@weights['w']
            maps={'quartic_raw':np.eye(len(keys)),'quartic_gaussian':np.linalg.solve(L.T,np.eye(len(keys))),**maps}
            rng=torch.Generator().manual_seed(20000+seed)
            xt=torch.randn(max(sizes),cfg['d'],generator=rng).numpy()
            xv=torch.randn(1024,cfg['d'],generator=rng).numpy()
            xe=torch.randn(4096,cfg['d'],generator=rng).numpy()
            phi=[design(x,keys) for x in [xt,xv,xe]]
            ys=[p@target for p in phi]
            error=max(float(np.max(np.abs(y-forward({'teacher_'+k:v for k,v in weights.items()},'teacher_',x)))) for x,y in zip([xt,xv,xe],ys))
            assert error<1e-9,error
            assert abs(target@H@target-1)<1e-10
            checks.append(dict(teacher=kind,seed=seed,max_direct_forward_error=error))
            evidence[f'{kind}_{seed}_target']=target
            for name,K in maps.items():
                evidence[f'{kind}_{seed}_{name}_map']=K
                feature=[p@K for p in phi]
                # The known Gaussian whitening is exact, not an estimated data-dependent transform.
                if name=='quartic_gaussian': np.testing.assert_allclose(K.T@H@K,np.eye(len(keys)),rtol=1e-11,atol=1e-11)
                for n in sizes:
                    Z=feature[0][:n];y=ys[0][:n]
                    U,s,Vh=np.linalg.svd(Z,full_matrices=False);uy=U.T@y
                    rank=int(np.sum(s>s[0]*1e-12))
                    betas=[];val=[]
                    for lam in grid:
                        filt=np.divide(1.,s,out=np.zeros_like(s),where=s>s[0]*1e-12) if lam==0 else s/(s*s+n*lam)
                        beta=Vh.T@(filt*uy);betas.append(beta)
                        val.append(float(np.mean((feature[1]@beta-ys[1])**2)))
                    for method,index in [('minnorm',0),('ridge_validation',int(np.argmin(val)))]:
                        beta=betas[index];c=K@beta;delta=c-target
                        evkey=f'{kind}_{seed}_{n}_{name}_{method}'
                        evidence[evkey]=c;evidence[evkey+'_beta']=beta
                        # First-order residual for this exact convex objective.
                        normal=Z.T@(Z@beta-y)/n+grid[index]*beta
                        residual=float(np.linalg.norm(normal)/max(1.,np.linalg.norm(Z.T@y/n)))
                        assert residual<1e-8,(evkey,residual)
                        rows.append(dict(teacher=kind,seed=seed,n=n,features=name,solver=method,
                                         distinct_fit_parameters=K.shape[1],design_rank=rank,lambda_selected=grid[index],
                                         train_mse=float(np.mean((Z@beta-y)**2)),validation_mse=val[index],
                                         test_mse=float(np.mean((feature[2]@beta-ys[2])**2)),
                                         exact_gaussian_mse=float(delta@H@delta),normal_equation_residual=residual,
                                         coefficient_key=evkey))
            print('Completed',kind,seed,flush=True)
    evidence['keys']=np.array(keys);evidence['moment_gram']=H
    np.savez_compressed(out/'evidence.npz',**evidence)
    write_csv(out/'results.csv',rows)
    (out/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    (out/'completed.json').write_text(json.dumps(dict(status='complete',rows=len(rows),evidence_sha256=hashlib.sha256((out/'evidence.npz').read_bytes()).hexdigest()))+'\n')


if __name__=='__main__': main()

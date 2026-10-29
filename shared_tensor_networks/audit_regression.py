"""Audit saved regression coefficients, direct oracle designs, and Gaussian moments."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch


def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);args=parser.parse_args()
    out=args.directory;protocol=json.loads((out/'protocol.json').read_text());src=Path(protocol['source_run'])
    cfg=json.loads((src/'protocol.json').read_text())['arguments'];sizes=[int(v) for v in cfg['ntrain'].split(',')]
    rows=list(csv.DictReader((out/'results.csv').open()))
    expected=len(cfg['teacher'].split(','))*cfg['seeds']*len(sizes)*4*2
    assert len(rows)==expected
    assert len({(r['teacher'],r['seed'],r['n'],r['features'],r['solver']) for r in rows})==expected
    ev=np.load(out/'evidence.npz');keys=ev['keys'];powers=keys[:,None,:]+keys[None,:,:]
    # Independent factorial expression, rather than the producer's odd-product routine.
    moments=np.array([0. if k%2 else math.factorial(k)/(2**(k//2)*math.factorial(k//2)) for k in range(9)])
    H=np.prod(moments[powers],axis=2)
    np.testing.assert_allclose(H,ev['moment_gram'],atol=0,rtol=0)
    max_design,max_loss,max_normal=0.,0.,0.;full_rank_recoveries=0;groups={}
    for kind in cfg['teacher'].split(','):
        for seed in range(cfg['seeds']):
            weights=np.load(src/f'{kind}_{seed}_{sizes[0]}_tied.npz')
            def W(k):return weights['teacher_'+k]
            rng=torch.Generator().manual_seed(20000+seed)
            x=torch.randn(max(sizes),cfg['d'],generator=rng,dtype=torch.float64).numpy()
            v=torch.randn(1024,cfg['d'],generator=rng,dtype=torch.float64).numpy()
            test=torch.randn(4096,cfg['d'],generator=rng,dtype=torch.float64).numpy()
            def monom(x):
                return np.column_stack([np.prod(x**k,axis=1) for k in keys])
            phis=[monom(a) for a in [x,v,test]]
            target=ev[f'{kind}_{seed}_target'];ys=[p@target for p in phis]
            left=(v@W('A').T)*(v@W('B').T)
            right=(v@W('AR').T)*(v@W('BR').T) if kind=='untied' else left
            hidden=np.column_stack([left[:,i]*right[:,j] for i in range(cfg['width']) for j in range(cfg['width']) if kind=='untied' or i<=j])
            top=(left@W('C').T)*(right@W('D').T)
            for name,direct in [('oracle_hidden',hidden),('oracle_top',top)]:
                predicted=phis[1]@ev[f'{kind}_{seed}_{name}_map']
                error=float(np.max(np.abs(direct-predicted))/max(1.,np.max(np.abs(direct))))
                assert error<1e-10;max_design=max(max_design,error)
            for row in [r for r in rows if r['teacher']==kind and int(r['seed'])==seed]:
                n=int(row['n']);K=ev[f'{kind}_{seed}_{row["features"]}_map'];key=row['coefficient_key']
                beta=ev[key+'_beta'];coeff=ev[key];np.testing.assert_allclose(K@beta,coeff,rtol=1e-10,atol=1e-10)
                delta=coeff-target;exact=float(delta@H@delta)
                denom=max(1.,abs(exact));error=abs(exact-float(row['exact_gaussian_mse']))/denom
                assert error<1e-10;max_loss=max(max_loss,error)
                for metric,phi,y in zip(['train_mse','validation_mse','test_mse'],[phis[0][:n],phis[1],phis[2]],[ys[0][:n],ys[1],ys[2]]):
                    value=float(np.mean((phi@coeff-y)**2));assert abs(value-float(row[metric]))<1e-9*max(1.,value)
                Z=phis[0][:n]@K;y=ys[0][:n];normal=Z.T@(Z@beta-y)/n+float(row['lambda_selected'])*beta
                residual=float(np.linalg.norm(normal)/max(1.,np.linalg.norm(Z.T@y/n)))
                assert residual<1e-8;max_normal=max(max_normal,residual)
                if row['solver']=='minnorm' and row['features']=='quartic_raw' and n>=len(keys):
                    assert int(row['design_rank'])==len(keys)
                    assert exact<1e-16,(key,exact)
                    full_rank_recoveries+=1
                group=(kind,seed,n,row['features']);groups.setdefault(group,{})[row['solver']]=row
    for group,r in groups.items():
        # Lambda=0 is included, so validation-selected ridge cannot be worse on validation.
        assert float(r['ridge_validation']['validation_mse'])<=float(r['minnorm']['validation_mse'])+1e-12
    result=dict(status='passed',rows=len(rows),full_rank_quartic_recoveries=full_rank_recoveries,
                max_direct_oracle_design_relative_error=max_design,max_scaled_population_error=max_loss,
                max_normal_equation_residual=max_normal,
                limitation='Recomputes selected solution metrics and stationarity, but does not independently replay every candidate in the ridge selection grid.',
                evidence_sha256=hashlib.sha256((out/'evidence.npz').read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()

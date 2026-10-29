"""Recompute paths with pivoted-QR least squares and direct ridge normal equations."""
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.linalg import lstsq

from experiment import Network, gram
from regression_baselines import feature_maps


def main():
    root=Path(__file__).parent;src=root/'results/e1_d8';out=root/'results/singular_feature_paths'
    rows=list(csv.DictReader((out/'results.csv').open()));assert len(rows)==350
    assert len({(r['teacher'],r['seed'],r['n'],r['epsilon']) for r in rows})==350
    keys=sorted({tuple(sum(i==j for i in inds) for j in range(8)) for inds in itertools.combinations_with_replacement(range(8),4)})
    L=np.linalg.cholesky(gram(keys));worst={k:0. for k in ['floor','coefficient_norm','ridge_error','ridge_norm']}
    cache={}
    for row in rows:
        kind,seed,n=row['teacher'],int(row['seed']),int(row['n']);eps=float(row['epsilon']);identity=(kind,seed,n)
        if identity not in cache:
            init={k:p.detach().numpy().copy() for k,p in Network(8,8,8,False,30000+seed).named_parameters()}
            init['AR']=init['A'].copy();init['BR']=init['B'].copy()
            with np.load(src/f'{kind}_{seed}_{n}_untied_width.npz') as data:
                trained={k.removeprefix('student_'):v.copy() for k,v in data.items() if k.startswith('student_')}
                teacher={k.removeprefix('teacher_'):v.copy() for k,v in data.items() if k.startswith('teacher_')}
            target=L.T@(feature_maps(teacher,8,keys)['oracle_top']@teacher['w']);cache[identity]=(init,trained,target)
        init,trained,target=cache[identity]
        weights={k:(1-eps)*init[k]+eps*trained[k] for k in init}
        W=L.T@feature_maps(weights,8,keys)['oracle_hidden'];W/=np.linalg.norm(W,axis=0)
        beta,_,rank,_=lstsq(W,target,cond=1e-10,lapack_driver='gelsy');assert rank==int(row['functional_rank'])
        br=np.linalg.solve(W.T@W+.001*np.eye(W.shape[1]),W.T@target)
        for label,value,field in [('floor',np.linalg.norm(W@beta-target)**2,'minimum_norm_floor'),
                                   ('coefficient_norm',np.linalg.norm(beta),'beta_norm'),
                                   ('ridge_error',np.linalg.norm(W@br-target)**2,'ridge_error'),
                                   ('ridge_norm',np.linalg.norm(br),'ridge_beta_norm')]:
            error=abs(value-float(row[field]))/max(1.,abs(value));assert error<1e-7,(identity,eps,label,error)
            worst[label]=max(worst[label],float(error))
    result=dict(status='passed',rows=350,method='pivoted QR (gelsy) for unregularized fits; linear solve for ridge',
                max_scaled_discrepancies=worst,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                results_sha256=hashlib.sha256((out/'results.csv').read_bytes()).hexdigest())
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()

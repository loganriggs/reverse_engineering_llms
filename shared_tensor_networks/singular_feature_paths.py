"""Conditioning of hidden product spans when untied branches leave equality."""
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from experiment import Network, gram, write_csv
from regression_baselines import feature_maps


def solve(W,t,lam):
    u,s,vh=np.linalg.svd(W,full_matrices=False)
    f=np.divide(1.,s,out=np.zeros_like(s),where=s>s[0]*1e-10) if lam==0 else s/(s*s+lam)
    beta=vh.T@(f*(u.T@t));residual=W@beta-t
    return beta,float(residual@residual),s


def main():
    root=Path(__file__).parent;src=root/'results/e1_d8';out=root/'results/singular_feature_paths'
    out.mkdir(parents=True,exist_ok=False)
    epsilons=[0.,1e-5,1e-4,1e-3,1e-2,.1,1.];lam=.001
    protocol=dict(epsilons=epsilons,population_ridge_lambda=lam,relative_rank_cutoff=1e-10,
                  paths='all 50 width-matched untied initial-to-600-step hidden-parameter straight lines',
                  scaling='each hidden product column normalized to unit Gaussian RMS before readout comparisons',
                  interpretation='parameter interpolation, not the optimization trajectory; truth-aware population diagnostics, not trained predictions',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  source_results_sha256=hashlib.sha256((src/'results.csv').read_bytes()).hexdigest())
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    toy=[]
    for eps in epsilons:
        W=np.array([[1.,1.],[0.,eps]]);target=np.array([0.,1.])
        b,e,s=solve(W,target,0.);br,er,_=solve(W,target,lam)
        if eps:
            np.testing.assert_allclose(b,[-1/eps,1/eps],rtol=1e-9,atol=1e-8)
            assert e<1e-15
        else:assert e==1
        np.testing.assert_allclose(br,np.linalg.solve(W.T@W+lam*np.eye(2),W.T@target),rtol=1e-10,atol=1e-10)
        toy.append(dict(epsilon=eps,minimum_norm_floor=e,beta_norm=float(np.linalg.norm(b)),ridge_error=er,ridge_beta_norm=float(np.linalg.norm(br))))
    keys=sorted({tuple(sum(i==j for i in inds) for j in range(8)) for inds in itertools.combinations_with_replacement(range(8),4)})
    L=np.linalg.cholesky(gram(keys));rows=[]
    for kind in ['tied','untied']:
        for seed in range(5):
            initial=Network(8,8,8,False,30000+seed)
            init={k:v.detach().numpy().copy() for k,v in initial.named_parameters()};init['AR']=init['A'].copy();init['BR']=init['B'].copy()
            for n in [64,128,256,512,1024]:
                with np.load(src/f'{kind}_{seed}_{n}_untied_width.npz') as data:
                    trained={k.removeprefix('student_'):v.copy() for k,v in data.items() if k.startswith('student_')}
                    teacher={k.removeprefix('teacher_'):v.copy() for k,v in data.items() if k.startswith('teacher_')}
                target=L.T@(feature_maps(teacher,8,keys)['oracle_top']@teacher['w'])
                for eps in epsilons:
                    weights={k:(1-eps)*init[k]+eps*trained[k] for k in init}
                    W=L.T@feature_maps(weights,8,keys)['oracle_hidden']
                    norms=np.linalg.norm(W,axis=0);assert np.all(norms>0);W=W/norms
                    b,e,s=solve(W,target,0.);br,er,_=solve(W,target,lam)
                    stationarity=float(np.linalg.norm(W.T@(W@br-target)+lam*br));assert stationarity<1e-8
                    rank=int(np.sum(s>s[0]*1e-10))
                    if eps==0:assert rank==36
                    rows.append(dict(teacher=kind,seed=seed,n=n,epsilon=eps,functional_rank=rank,
                                     largest_singular_value=float(s[0]),smallest_singular_value=float(s[-1]),
                                     minimum_norm_floor=e,beta_norm=float(np.linalg.norm(b)),
                                     ridge_error=er,ridge_beta_norm=float(np.linalg.norm(br)),ridge_stationarity=stationarity))
            print('Completed',kind,seed,flush=True)
    write_csv(out/'results.csv',rows)
    (out/'toy_checks.json').write_text(json.dumps(toy,indent=2)+'\n')
    (out/'completed.json').write_text(json.dumps(dict(status='complete',rows=len(rows),toy_cases=len(toy)))+'\n')


if __name__=='__main__':main()

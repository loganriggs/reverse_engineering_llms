"""Noise and repeat-count red team for a fixed-function edit-sparse gauge."""

import json

import numpy as np

from dual_joint_function_canonical import HERE
from shared_pair_dense_edit_gauge import PAIRS,TRUTH,fit_changes
from shared_pair_gauge import profile

SEEDS=(0,1,2)
REPS=10
SIGMAS=(0.,.2,.5,1.)
OBSERVATIONS=(1,4,16,64,256)
BASE_SEED=20261106


def run():
    analysis=json.loads((HERE/'shared_pair_dense_analysis.json').read_text())
    edges=np.asarray(PAIRS['all_pairs'],dtype=int)
    rows=[]
    for seed in SEEDS:
        source=next(x for x in analysis['rows'] if x['seed']==seed)
        matrix=np.asarray(source['matrices']['uniform_rank3'])
        q=np.asarray(source['quadratic_effective'])
        _,s,v=np.linalg.svd(matrix,full_matrices=False)
        assert s[3]<1e-7
        base=v[:3]
        ideal=base[:,edges[:,1]]-base[:,edges[:,0]]
        scale=float(np.median(np.linalg.norm(ideal,axis=0)))
        for rep in range(REPS):
            for sigma in SIGMAS:
                noise_seed=BASE_SEED+seed*10000+rep*100+int(sigma*10)
                rng=np.random.default_rng(noise_seed)
                iid=rng.normal(size=(max(OBSERVATIONS),)+ideal.shape)
                fixed=rng.normal(size=ideal.shape)
                for mode in ('iid','fixed_pair_bias'):
                    cached=None
                    for n in OBSERVATIONS:
                        if cached is None or (mode=='iid' and sigma>0):
                            error=(iid[:n].mean(axis=0) if mode=='iid' else fixed)
                            observed=ideal+sigma*scale*error
                            fit_seed=BASE_SEED+500000+seed*10000+rep*100+int(sigma*10)+(
                                0 if mode=='iid' else 50)
                            d,loss=fit_changes(base,observed,fit_seed)
                            assert np.linalg.norm(matrix@d.T@d-matrix)/np.linalg.norm(matrix)<1e-8
                            p=profile(matrix,q,d,TRUTH)
                            cached=(loss,p)
                        else:
                            loss,p=cached
                        rows.append(dict(seed=seed,rep=rep,sigma=sigma,
                            noise_mode=mode,n_observations_per_pair=n,
                            noise_seed=noise_seed,noise_scale=scale,
                            fit_edit_l1=loss,
                            matched_variable_cosine=p['matched_variable_cosine'],
                            mean_cell_top_one_monomial_energy_fraction=
                                p['mean_cell_top_one_monomial_energy_fraction']))
        print('seed',seed,flush=True)
    summary=[]
    for sigma in SIGMAS:
        for mode in ('iid','fixed_pair_bias'):
            for n in OBSERVATIONS:
                part=[r for r in rows if r['sigma']==sigma and r['noise_mode']==mode
                      and r['n_observations_per_pair']==n]
                align=np.asarray([r['matched_variable_cosine'] for r in part])
                cell=np.asarray([r['mean_cell_top_one_monomial_energy_fraction']
                                 for r in part])
                summary.append(dict(sigma=sigma,noise_mode=mode,
                    n_observations_per_pair=n,n=len(part),
                    mean_alignment=float(align.mean()),
                    fraction_alignment_above_0_98=float(np.mean(align>.98)),
                    mean_cell_top_one_fraction=float(cell.mean()),
                    fraction_cell_top_one_above_0_95=float(np.mean(cell>.95))))
    result=dict(description='Complete 28-pair edit ensemble with Gaussian measurement noise; iid observations average down, fixed per-pair bias does not; 3 trained dense-code models',
        seeds=SEEDS,reps=REPS,sigmas=SIGMAS,observations=OBSERVATIONS,
        base_seed=BASE_SEED,noise_scale='coordinate standard deviation = sigma times median exact pair-delta L2',
        caveat='Noise is synthetic and injected into the gauge-fitting edit vectors, not model activations; no task-pruning evaluation on noisy fits.',
        summary=summary,rows=rows)
    (HERE/'shared_pair_dense_edit_noise.json').write_text(json.dumps(result,indent=2)+'\n')
    for row in summary:print(row,flush=True)


if __name__=='__main__':run()

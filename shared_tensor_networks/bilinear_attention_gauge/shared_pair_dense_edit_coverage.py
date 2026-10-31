"""Finite random-edit coverage for the dense-token exact-gauge prior."""

import argparse
import json

import numpy as np

from dual_joint_function_canonical import HERE
from shared_pair_dense_edit_gauge import PAIRS,TRUTH,fit_edit
from shared_pair_gauge import profile

SEEDS=(0,1,2)
SAMPLE_SIZES=(2,4,8,12,16,28,56)
REPS=30
RNG_SEED=20261105


def run(scheme='with_replacement'):
    assert scheme in ('with_replacement','without_replacement')
    rng=np.random.default_rng(RNG_SEED)
    analysis=json.loads((HERE/'shared_pair_dense_analysis.json').read_text())
    all_pairs=PAIRS['all_pairs']
    rows=[]
    for seed in SEEDS:
        source=next(r for r in analysis['rows'] if r['seed']==seed)
        matrix=np.asarray(source['matrices']['uniform_rank3'])
        q=np.asarray(source['quadratic_effective'])
        _,s,v=np.linalg.svd(matrix,full_matrices=False)
        assert s[3]<1e-7
        base=v[:3]
        sizes=SAMPLE_SIZES if scheme=='with_replacement' else (2,4,8,12,16,20,24,28)
        for n in sizes:
            for rep in range(REPS):
                indices=rng.choice(len(all_pairs),size=n,
                                   replace=(scheme=='with_replacement'))
                sampled=[all_pairs[i] for i in indices]
                d,objective=fit_edit(base,sampled,RNG_SEED+seed*10000+n*100+rep)
                result=profile(matrix,q,d,TRUTH)
                unseen=[pair for pair in all_pairs if pair not in set(sampled)]
                heldout=float(np.mean([np.sum(np.abs(d[:,j]-d[:,i]))
                    for i,j in unseen])) if unseen else None
                rows.append(dict(seed=seed,n_samples=n,rep=rep,
                    sampled_pairs=[list(pair) for pair in sampled],
                    unique_pairs=len(set(sampled)),unique_tokens=len(set(
                        token for pair in sampled for token in pair)),
                    fit_edit_l1=objective,heldout_edit_l1=heldout,
                    matched_variable_cosine=result['matched_variable_cosine'],
                    mean_cell_top_one_monomial_energy_fraction=
                        result['mean_cell_top_one_monomial_energy_fraction']))
        print('seed',seed,flush=True)
    summary=[]
    for n in sizes:
        part=[r for r in rows if r['n_samples']==n]
        align=np.array([r['matched_variable_cosine'] for r in part])
        cell=np.array([r['mean_cell_top_one_monomial_energy_fraction'] for r in part])
        summary.append(dict(n_samples=n,n=len(part),
            mean_unique_pairs=float(np.mean([r['unique_pairs'] for r in part])),
            mean_unique_tokens=float(np.mean([r['unique_tokens'] for r in part])),
            mean_alignment=float(np.mean(align)),
            fraction_alignment_above_0_98=float(np.mean(align>.98)),
            mean_cell_top_one_fraction=float(np.mean(cell)),
            fraction_cell_top_one_above_0_95=float(np.mean(cell>.95))))
    result=dict(description='Random token swap coverage on three trained dense-token models; edit-fit labels absent, known variables used only after fitting',
        scheme=scheme,seeds=SEEDS,sample_sizes=sizes,reps=REPS,rng_seed=RNG_SEED,
        summary=summary,rows=rows)
    name=('shared_pair_dense_edit_coverage.json' if scheme=='with_replacement' else
          'shared_pair_dense_edit_diverse.json')
    (HERE/name).write_text(json.dumps(result,indent=2)+'\n')
    for row in summary:print(row,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--scheme',choices=('with_replacement','without_replacement'),
                   default='with_replacement')
    run(p.parse_args().scheme)

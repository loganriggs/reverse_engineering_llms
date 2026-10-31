"""Predeclared 8k-context, split, trim, and external-corpus unlabeled red team."""

import json

import numpy as np

from flagship_later_head_gauge_capture import TARGETS
from flagship_later_head_unlabeled_gauge import fit,pca,sparse_axis
from flagship_transfer import ART

SEED=20261104
METHODS=('pca_centered','activation_l1_raw','activation_l1_centered','varimax_centered')


def run():
    rng=np.random.default_rng(SEED)
    a=np.load(ART/'later_head_unlabeled_large.npz')
    d=np.load(ART/'later_head_gauge_capture.npz')
    m=json.loads((ART/'later_head_gauge_capture.json').read_text())
    existing=np.load(ART/'later_head_exact_gauge_rotations.npz')
    selection=json.loads((ART/'later_head_exact_gauge_finite.json').read_text())['summary']
    records=[];rotations={}
    for target in TARGETS:
        train=a[f'{target}__fineweb_train'].astype(np.float64)
        norms=np.linalg.norm(train-train.mean(axis=0),axis=1)
        threshold=np.quantile(norms,.998)
        panels={'full_8192':train,'first_4096':train[:4096],
                'second_4096':train[4096:],
                'trim_top_0p2_percent':train[norms<=threshold],
                'fineweb_test_2048':a[f'{target}__fineweb_test'].astype(np.float64),
                'wikitext103_1024':a[f'{target}__wikitext103'].astype(np.float64),
                'tinystories_1024':a[f'{target}__tinystories'].astype(np.float64)}
        delta=d[f'{target}__holdout__d'].astype(np.float64)
        grad=d[f'{target}__holdout__g'].astype(np.float64)
        fam=np.array([x['family'] for x in m['rows'][f'{target}__holdout']])
        own=fam==target;full=(delta*grad).sum(axis=1)
        supervised=existing[f'{target}__effect__all'][:,selection[f'{target}__effect__all']['selected_axis']]
        for panel,z in panels.items():
            for method in METHODS:
                fit_z=z-z.mean(axis=0) if method.endswith('centered') else z
                if method.startswith('pca'):
                    r=pca(fit_z)
                else:
                    kind='varimax' if method.startswith('varimax') else 'activation_l1'
                    r=fit(fit_z,kind,rng)[0][1]
                axis,ratios=sparse_axis(z,r)
                response=(delta@r)*(grad@r)
                selected=response[:,axis]
                oracle=min(np.mean(np.abs(response[own,j]-full[own])) for j in range(4))/np.mean(np.abs(full[own]))
                records.append(dict(target=target,panel=panel,nfit=len(z),method=method,
                    selected_axis=axis,train_sparsity_ratios=ratios,
                    selected_axis_cosine_to_supervised=float(abs(r[:,axis]@supervised)),
                    holdout_target_linear_relative_mae=float(
                        np.mean(np.abs(selected[own]-full[own]))/np.mean(np.abs(full[own]))),
                    holdout_target_oracle_relative_mae=float(oracle)))
                rotations[f'{target}__{panel}__{method}']=r
                print(target,panel,method,'rel',round(records[-1]['holdout_target_linear_relative_mae'],3),flush=True)
    result=dict(description='Unlabeled natural-data 8192-context stability and 0.2% norm-tail trimming; task labels only evaluate frozen axes',
                seed=SEED,methods=METHODS,records=records)
    (ART/'later_head_unlabeled_large_eval.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez(ART/'later_head_unlabeled_large_eval_rotations.npz',**rotations)
    print('Saved large evaluation',flush=True)


if __name__=='__main__':run()

"""Independent-split and cross-corpus stability of unlabeled head gauges."""

import json

import numpy as np

from flagship_later_head_exact_gauge import effective_axes
from flagship_later_head_gauge_capture import TARGETS
from flagship_later_head_unlabeled_gauge import fit,pca,sparse_axis
from flagship_transfer import ART

SEED=20261103
METHODS=('pca_centered','activation_l1_raw','activation_l1_centered','varimax_centered')


def run():
    rng=np.random.default_rng(SEED)
    a=np.load(ART/'later_head_unlabeled_activations.npz')
    d=np.load(ART/'later_head_gauge_capture.npz')
    m=json.loads((ART/'later_head_gauge_capture.json').read_text())
    originals=np.load(ART/'later_head_unlabeled_gauge_rotations.npz')
    reference=json.loads((ART/'later_head_unlabeled_gauge.json').read_text())['records']
    records=[];rotations={}
    for target in TARGETS:
        delta=d[f'{target}__holdout__d'].astype(np.float64)
        grad=d[f'{target}__holdout__g'].astype(np.float64)
        families=np.array([x['family'] for x in m['rows'][f'{target}__holdout']])
        own=families==target
        full=np.sum(delta*grad,axis=-1)
        for length in (5,32):
            base=a[f'{target}__fineweb_train__l{length}'].astype(np.float64)
            panels={'first_half':base[:256],'second_half':base[256:],
                    'fineweb_test':a[f'{target}__fineweb_test__l{length}'].astype(np.float64),
                    'wikitext103':a[f'{target}__wikitext103__l{length}'].astype(np.float64),
                    'tinystories':a[f'{target}__tinystories__l{length}'].astype(np.float64)}
            for method in METHODS:
                original=next(x for x in reference if x['target']==target and x['length']==length and x['method']==method)
                original_r=originals[f'{target}__l{length}__{method}']
                original_u=original_r[:,original['selected_axis']]
                for panel,z in panels.items():
                    train=z-z.mean(axis=0) if method.endswith('centered') else z
                    if method.startswith('pca'):
                        r=pca(train)
                    else:
                        kind='varimax' if method.startswith('varimax') else 'activation_l1'
                        r=fit(train,kind,rng)[0][1]
                    axis,ratios=sparse_axis(z,r)
                    key=f'{target}__l{length}__{method}__{panel}'
                    rotations[key]=r
                    effect=(delta@r)*(grad@r)
                    selected=effect[:,axis]
                    records.append(dict(target=target,length=length,method=method,panel=panel,
                        nfit=len(z),selected_axis=axis,train_sparsity_ratios=ratios,
                        selected_axis_cosine_to_full_train=float(abs(r[:,axis]@original_u)),
                        holdout_target_linear_relative_mae=float(
                            np.mean(np.abs(selected[own]-full[own]))/np.mean(np.abs(full[own]))),
                        holdout_target_effective_axes=float(np.mean(effective_axes(effect[own])))))
                    print(target,length,method,panel,'relative',
                          round(records[-1]['holdout_target_linear_relative_mae'],3),flush=True)
    result=dict(description='Unlabeled natural-data split and corpus refits; target labels used only for heldout diagnostic',
                seed=SEED,methods=METHODS,records=records)
    (ART/'later_head_unlabeled_stability.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez(ART/'later_head_unlabeled_stability_rotations.npz',**rotations)
    print('Saved stability',flush=True)


if __name__=='__main__':run()

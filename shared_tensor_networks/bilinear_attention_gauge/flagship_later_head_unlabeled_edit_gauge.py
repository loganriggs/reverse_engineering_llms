"""Exact gauges from balanced answer-free token-substitution responses."""

import json

import numpy as np

from flagship_later_head_exact_gauge import effective_axes,normalized_rows
from flagship_later_head_gauge_capture import TARGETS
from flagship_later_head_unlabeled_gauge import fit,pca,sparse_axis
from flagship_transfer import ART

SEED=20261112
TYPE_FOR_TARGET={'bracket_closure':'bracket','subject_verb_agreement':'noun_number'}
METHODS=('raw_pca','normalized_pca','edit_l1')


def fit_basis(data,method,rng):
    if method=='raw_pca':return pca(data)
    if method=='normalized_pca':return pca(normalized_rows(data))
    if method=='edit_l1':return fit(data,'activation_l1',rng)[0][1]
    raise ValueError(method)


def run():
    rng=np.random.default_rng(SEED)
    edits=np.load(ART/'later_head_unlabeled_edits.npz')
    meta=json.loads((ART/'later_head_unlabeled_edits.json').read_text())
    task=np.load(ART/'later_head_gauge_capture.npz')
    task_meta=json.loads((ART/'later_head_gauge_capture.json').read_text())
    records=[];rotations={}
    for target in TARGETS:
        own_type=TYPE_FOR_TARGET[target]
        d_task=task[f'{target}__holdout__d'].astype(np.float64)
        g_task=task[f'{target}__holdout__g'].astype(np.float64)
        own_task=np.array([r['family']==target for r in task_meta['rows'][f'{target}__holdout']])
        full=(d_task*g_task).sum(axis=1)
        val_types=np.array([r['type'] for r in meta['panels']['validation']])
        val_d=edits[f'{target}__validation'].astype(np.float64)
        for panel in ('train','validation'):
            data=edits[f'{target}__{panel}'].astype(np.float64)
            types=np.array([r['type'] for r in meta['panels'][panel]])
            for condition in ('all','exclude_target_type','only_target_type'):
                mask=(np.ones(len(data),bool) if condition=='all' else
                      types!=own_type if condition=='exclude_target_type' else
                      types==own_type)
                fit_d=data[mask]
                for method in METHODS:
                    r=fit_basis(fit_d,method,rng)
                    key=f'{target}__{panel}__{condition}__{method}'
                    rotations[key]=r
                    projected=fit_d@r
                    selected={'global_energy':int(np.argmax(np.mean(projected**2,axis=0))),
                              'rare_coordinate':sparse_axis(fit_d,r)[0]}
                    task_response=(d_task@r)*(g_task@r)
                    for rule,axis in selected.items():
                        predicted=task_response[:,axis]
                        val_coord=np.abs(val_d@r[:,axis])
                        own_mean=float(np.mean(val_coord[val_types==own_type]))
                        other_mean=float(np.mean(val_coord[val_types!=own_type]))
                        records.append(dict(target=target,panel=panel,condition=condition,
                            method=method,selection_rule=rule,fit_count=int(mask.sum()),
                            selected_axis=axis,
                            holdout_task_effective_axes=float(np.mean(effective_axes(task_response[own_task]))),
                            holdout_task_axis_local_relative_mae=float(
                                np.mean(np.abs(predicted[own_task]-full[own_task]))/
                                np.mean(np.abs(full[own_task]))),
                            validation_own_edit_abs_coordinate=own_mean,
                            validation_other_edit_abs_coordinate=other_mean,
                            validation_own_edit_enrichment=own_mean/max(other_mean,1e-12)))
                    print(target,panel,condition,method,'global',
                          round(records[-2]['holdout_task_axis_local_relative_mae'],3),
                          'rare',round(records[-1]['holdout_task_axis_local_relative_mae'],3),flush=True)
    result=dict(description='Exact rank-4 head rotations fitted to answer-free token substitutions; task gradients only evaluate',
                seed=SEED,methods=METHODS,selection_rules=['global_energy','rare_coordinate'],
                type_for_target=TYPE_FOR_TARGET,records=records,
                caveat='Only-target-type condition uses input-edit categories but no output answers. Exclude-target-type is the whole-edit-type holdout.')
    (ART/'later_head_unlabeled_edit_gauge.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez(ART/'later_head_unlabeled_edit_gauge_rotations.npz',**rotations)
    print('Saved edit gauge',flush=True)


if __name__=='__main__':run()

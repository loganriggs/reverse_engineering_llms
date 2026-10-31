"""Random-axis controls and edit-type enrichment for answer-free gauges."""

import json

import numpy as np

from flagship_later_head_gauge_capture import TARGETS
from flagship_transfer import ART

TYPE_FOR_TARGET={'bracket_closure':'bracket','subject_verb_agreement':'noun_number'}


def run():
    e=np.load(ART/'later_head_unlabeled_edits.npz')
    meta=json.loads((ART/'later_head_unlabeled_edits.json').read_text())
    task=np.load(ART/'later_head_gauge_capture.npz')
    task_meta=json.loads((ART/'later_head_gauge_capture.json').read_text())
    r=np.load(ART/'later_head_unlabeled_edit_gauge_rotations.npz')
    rows=json.loads((ART/'later_head_unlabeled_edit_gauge.json').read_text())['records']
    random=np.load(ART/'later_head_exact_gauge_rotations.npz')
    result={}
    for target in TARGETS:
        own_type=TYPE_FOR_TARGET[target]
        d=task[f'{target}__holdout__d'];g=task[f'{target}__holdout__g']
        own=np.array([x['family']==target for x in task_meta['rows'][f'{target}__holdout']])
        full=(d*g).sum(1)
        val=e[f'{target}__validation']
        val_type=np.array([x['type'] for x in meta['panels']['validation']])
        target_rows={}
        for panel in ('train','validation'):
            train=e[f'{target}__{panel}']
            null=[]
            for i in range(64):
                z=random[f'{target}__random_{i}']
                axis=int(np.argmax(np.mean((train@z)**2,axis=0)))
                pred=(d@z[:,axis])*(g@z[:,axis])
                null.append(float(np.mean(np.abs(pred[own]-full[own]))/
                                  np.mean(np.abs(full[own]))))
            fitted={}
            for method in ('raw_pca','normalized_pca','edit_l1'):
                row=next(x for x in rows if x['target']==target and x['panel']==panel and
                         x['condition']=='all' and x['method']==method and
                         x['selection_rule']=='global_energy')
                z=r[f'{target}__{panel}__all__{method}']
                axis=row['selected_axis']
                score=np.abs(val@z[:,axis])
                top=np.argsort(score)[-80:]
                fitted[method]=dict(axis=axis,
                    target_local_relative_mae=row['holdout_task_axis_local_relative_mae'],
                    empirical_random_rank=(1+sum(v<=row['holdout_task_axis_local_relative_mae'] for v in null))/65,
                    validation_top_decile_target_precision=float(np.mean(val_type[top]==own_type)),
                    validation_top_decile_target_recall=float(np.sum(val_type[top]==own_type)/80),
                    validation_edit_type_enrichment=row['validation_own_edit_enrichment'])
            target_rows[panel]=dict(random_median=float(np.median(null)),
                                    random_min=float(np.min(null)),fitted=fitted)
        cosine={}
        for method in ('raw_pca','normalized_pca','edit_l1'):
            selected=[]
            for panel in ('train','validation'):
                key=f'{target}__{panel}__all__{method}'
                z=r[key];axis=target_rows[panel]['fitted'][method]['axis']
                selected.append(z[:,axis])
            cosine[method]=float(abs(selected[0]@selected[1]))
        val_norm=np.linalg.norm(val,axis=1)
        top=np.argsort(val_norm)[-80:]
        result[target]=dict(panels=target_rows,
            train_validation_axis_cosine=cosine,
            validation_full_head_norm_top_decile_target_precision=float(
                np.mean(val_type[top]==own_type)),
            validation_mean_delta_norm_by_type={kind:float(np.mean(
                np.linalg.norm(val[val_type==kind],axis=1)))
                for kind in sorted(set(val_type))})
    (ART/'later_head_unlabeled_edit_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result,flush=True)


if __name__=='__main__':run()

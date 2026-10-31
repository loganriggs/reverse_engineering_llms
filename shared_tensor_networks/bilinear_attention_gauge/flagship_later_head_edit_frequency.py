"""Resampled edit-frequency threshold for answer-free raw-PCA head axes."""

import json

import numpy as np

from flagship_later_head_gauge_capture import TARGETS
from flagship_later_head_unlabeled_edit_gauge import TYPE_FOR_TARGET
from flagship_transfer import ART

SEED=20261113
PREVALENCE=(0.,.0001,.001,.005,.01,.05,.1)
SAMPLE_SIZES=(800,8000)
REPLICATES=100


def run():
    rng=np.random.default_rng(SEED)
    e=np.load(ART/'later_head_unlabeled_edits.npz')
    meta=json.loads((ART/'later_head_unlabeled_edits.json').read_text())
    task=np.load(ART/'later_head_gauge_capture.npz')
    task_meta=json.loads((ART/'later_head_gauge_capture.json').read_text())
    kinds=np.array([x['type'] for x in meta['panels']['train']])
    records=[]
    for target in TARGETS:
        own_type=TYPE_FOR_TARGET[target]
        data=e[f'{target}__train'].astype(np.float64)
        own_pool=np.where(kinds==own_type)[0]
        other_pool=np.where(kinds!=own_type)[0]
        d=task[f'{target}__holdout__d'].astype(np.float64)
        g=task[f'{target}__holdout__g'].astype(np.float64)
        own=np.array([x['family']==target for x in task_meta['rows'][f'{target}__holdout']])
        full=(d*g).sum(1)
        for n in SAMPLE_SIZES:
            for p in PREVALENCE:
                errors=[];observed=[]
                for _ in range(REPLICATES):
                    hit=int(rng.binomial(n,p))
                    chosen=np.concatenate((rng.choice(own_pool,hit,replace=True),
                                           rng.choice(other_pool,n-hit,replace=True)))
                    covariance=data[chosen].T@data[chosen]/n
                    eig,vec=np.linalg.eigh(covariance)
                    axis=vec[:,np.argmax(eig)]
                    response=(d@axis)*(g@axis)
                    errors.append(float(np.mean(np.abs(response[own]-full[own]))/
                                        np.mean(np.abs(full[own]))))
                    observed.append(hit)
                records.append(dict(target=target,n=n,target_type_probability=p,
                    mean_target_edits=float(np.mean(observed)),
                    zero_target_edits=int(np.sum(np.array(observed)==0)),
                    median_local_relative_mae=float(np.median(errors)),
                    p90_local_relative_mae=float(np.percentile(errors,90)),
                    fraction_below_10_percent=float(np.mean(np.array(errors)<.1)),
                    fraction_below_25_percent=float(np.mean(np.array(errors)<.25))))
                print(target,n,p,'median',round(records[-1]['median_local_relative_mae'],3),
                      'success',records[-1]['fraction_below_10_percent'],flush=True)
    result=dict(description='Bootstrap from fixed 80-per-type answer-free edit pool; raw-PCA top direction under target-edit prevalence sweep',
                seed=SEED,replicates=REPLICATES,sample_sizes=SAMPLE_SIZES,
                prevalence=PREVALENCE,success_rule='heldout task local path relative MAE < .1',
                caveat='Resampling a fixed intervention pool; not new model forwards. Type prevalence is a declared design measure, not inferred natural frequency.',
                records=records)
    (ART/'later_head_edit_frequency.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':run()

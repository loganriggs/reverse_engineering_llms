"""Remove highest-norm short FineWeb samples before unlabeled bracket fitting."""

import json

import numpy as np
from tokenizers import Tokenizer

from flagship_explicit_task_screen import TOKENIZER
from flagship_later_head_unlabeled_gauge import fit,sparse_axis
from flagship_transfer import ART

TARGET='bracket_closure'


def run():
    a=np.load(ART/'later_head_unlabeled_activations.npz')
    d=np.load(ART/'later_head_gauge_capture.npz')
    meta=json.loads((ART/'later_head_gauge_capture.json').read_text())
    z=a[f'{TARGET}__fineweb_train__l5'].astype(np.float64)
    ids=a['ids__fineweb_train__l5']
    norm=np.linalg.norm(z-z.mean(axis=0),axis=1)
    order=np.argsort(norm)[::-1]
    tokenizer=Tokenizer.from_file(TOKENIZER)
    delta=d[f'{TARGET}__holdout__d'].astype(np.float64)
    grad=d[f'{TARGET}__holdout__g'].astype(np.float64)
    own=np.array([x['family']==TARGET for x in meta['rows'][f'{TARGET}__holdout']])
    full=(delta*grad).sum(axis=1)
    rows=[]
    for remove in (0,1,2,5,10,20):
        keep=np.ones(len(z),dtype=bool);keep[order[:remove]]=False
        fit_z=z[keep]
        r=fit(fit_z-fit_z.mean(axis=0),'activation_l1',
              np.random.default_rng(100+remove))[0][1]
        axis,_=sparse_axis(fit_z,r)
        selected=(delta@r[:,axis])*(grad@r[:,axis])
        rows.append(dict(removed=remove,nfit=len(fit_z),axis=axis,
            target_local_relative_mae=float(
                np.mean(np.abs(selected[own]-full[own]))/np.mean(np.abs(full[own])))))
    top=[dict(index=int(i),norm=float(norm[i]),
              text=tokenizer.decode(ids[i].tolist())) for i in order[:10]]
    result=dict(description='Influence of highest-norm unlabeled natural contexts on 512-example bracket-axis fit',
                rows=rows,top_contexts=top)
    (ART/'later_head_unlabeled_outlier.json').write_text(json.dumps(result,indent=2)+'\n')
    print(rows,flush=True)


if __name__=='__main__':run()

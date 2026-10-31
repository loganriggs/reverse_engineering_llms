"""Balanced cross-corpus unlabeled prior with independent split replication."""

import json

import numpy as np

from flagship_later_head_gauge_capture import TARGETS
from flagship_later_head_unlabeled_gauge import fit,pca,sparse_axis
from flagship_transfer import ART

SEED=20261105
METHODS=('pca_centered','activation_l1_centered','varimax_centered')


def ratios(z,r):
    y=z@r;y=y-y.mean(0)
    return np.mean(np.abs(y),axis=0)/np.sqrt(np.mean(y*y,axis=0)).clip(1e-12)


def run():
    rng=np.random.default_rng(SEED)
    a=np.load(ART/'later_head_unlabeled_large.npz')
    d=np.load(ART/'later_head_gauge_capture.npz')
    meta=json.loads((ART/'later_head_gauge_capture.json').read_text())
    supervised=np.load(ART/'later_head_exact_gauge_rotations.npz')
    selected=json.loads((ART/'later_head_exact_gauge_finite.json').read_text())['summary']
    records=[];rotations={}
    for target in TARGETS:
        fine=a[f'{target}__fineweb_train'].astype(np.float64)
        wiki=a[f'{target}__wikitext103'].astype(np.float64)
        tiny=a[f'{target}__tinystories'].astype(np.float64)
        delta=d[f'{target}__holdout__d'].astype(np.float64)
        grad=d[f'{target}__holdout__g'].astype(np.float64)
        fam=np.array([x['family'] for x in meta['rows'][f'{target}__holdout']])
        own=fam==target;full=np.sum(delta*grad,axis=1)
        gold=supervised[f'{target}__effect__all'][:,selected[f'{target}__effect__all']['selected_axis']]
        for split in (0,1):
            natural={
                'fineweb':fine[split*4096:(split+1)*4096][rng.choice(4096,512,replace=False)],
                'wikitext103':wiki[split*512:(split+1)*512],
                'tinystories':tiny[split*512:(split+1)*512]}
            centered=[z-z.mean(0) for z in natural.values()]
            pool=np.concatenate(centered)
            for method in METHODS:
                if method=='pca_centered':r=pca(pool)
                else:
                    kind='varimax' if method.startswith('varimax') else 'activation_l1'
                    r=fit(pool,kind,rng)[0][1]
                rotations[f'{target}__split{split}__{method}']=r
                score=np.stack([ratios(z,r) for z in natural.values()])
                selection={'pooled_rare':int(np.argmin(score.mean(axis=0))),
                           'worst_corpus_rare':int(np.argmin(score.max(axis=0)))}
                response=(delta@r)*(grad@r)
                for rule,axis in selection.items():
                    value=response[:,axis]
                    records.append(dict(target=target,split=split,method=method,rule=rule,
                        selected_axis=axis,train_corpus_ratios=score.tolist(),
                        selected_axis_cosine_to_supervised=float(abs(r[:,axis]@gold)),
                        target_linear_relative_mae=float(
                            np.mean(np.abs(value[own]-full[own]))/np.mean(np.abs(full[own]))),
                        target_oracle_relative_mae=float(min(
                            np.mean(np.abs(response[own,j]-full[own]))/np.mean(np.abs(full[own]))
                            for j in range(4)))))
                    print(target,split,method,rule,'relative',
                          round(records[-1]['target_linear_relative_mae'],3),flush=True)
    result=dict(description='Balanced FineWeb/WikiText/TinyStories unlabeled rotation; two independent natural-data splits; target labels only evaluate',
                seed=SEED,fit_contexts_per_corpus_per_split=512,
                methods=METHODS,selection_rules=['pooled_rare','worst_corpus_rare'],
                records=records)
    (ART/'later_head_multicorpus_unlabeled.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez(ART/'later_head_multicorpus_unlabeled_rotations.npz',**rotations)
    print('Saved multicorpus',flush=True)


if __name__=='__main__':run()

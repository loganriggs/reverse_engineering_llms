"""Document-level uncertainty for a balanced full-model sensitivity gauge."""

import json

import numpy as np
from scipy.optimize import linear_sum_assignment

from flagship_component_labels import CATEGORIES
from flagship_transfer import ART


STEM='all_head_balanced_output_gauge_train16_val8_test16_ext8_s9171'


def coordinates(vectors, rotation):
    x=vectors @ rotation
    return np.sum(np.abs(x),axis=1)**2 / np.maximum(np.sum(x*x,axis=1),1e-30)


def bootstrap_difference(a,b,rng,n=10000):
    diff=np.asarray(a)-np.asarray(b)
    indices=rng.integers(0,len(diff),(n,len(diff)))
    means=diff[indices].mean(1)
    return {'mean':float(diff.mean()),
            'ci95':[float(v) for v in np.quantile(means,[.025,.975])],
            'n_documents':len(diff),
            'fraction_documents_improved':float(np.mean(diff<0))}


def category_axis_transfer(arrays,meta,rotation,mode):
    fit=meta['fit_categories']
    def energy(panel):
        x=arrays[f'{panel}_vectors'].reshape(-1,meta['length'],256)
        cats=arrays[f'{panel}_categories'].reshape(-1,meta['length'])[:,0]
        if mode=='fixed_key_position':x=x[:,meta['length']-2:meta['length']-1]
        z=x@rotation
        value=np.mean(z*z,axis=1)
        value/=np.maximum(value.sum(1,keepdims=True),1e-30)
        return value,cats
    train,train_cats=energy('train_fineweb')
    results={}
    for category in fit:
        cid=CATEGORIES.index(category)
        others=np.isin(train_cats,[CATEGORIES.index(x) for x in fit if x!=category])
        numerator=train[train_cats==cid].mean(0)
        denominator=train[others].mean(0)
        axis=int(np.argmax((numerator+1e-5)/(denominator+1e-5)))
        transfer={}
        for panel in meta['splits']:
            e,cats=energy(panel)
            other=np.isin(cats,[CATEGORIES.index(x) for x in fit if x!=category])
            if not np.any(cats==cid) or not np.any(other):continue
            enrichment=float((e[cats==cid,axis].mean()+1e-5)/
                             (e[other,axis].mean()+1e-5))
            transfer[panel]={'enrichment':enrichment,
                             'n_category_documents':int(np.sum(cats==cid)),
                             'n_other_fit_documents':int(np.sum(other))}
        results[category]={'axis':axis,'transfer':transfer}
    return results


def run(stem=STEM):
    meta=json.loads((ART/f'{stem}.json').read_text())
    arrays=np.load(ART/f'{stem}.npz')
    rotations=np.load(ART/f'{stem}_rotations.npz')
    choices=('native','weight_0','response_0','balanced_fit_categories',
             'pooled_all_categories_oracle','held_categories_oracle')
    rng=np.random.default_rng(92871)
    output={}
    for panel in meta['splits']:
        vectors=arrays[f'{panel}_vectors'].astype(np.float64)
        cats=arrays[f'{panel}_categories']
        docs=arrays[f'{panel}_document_ids']
        scores={name:coordinates(vectors,rotations[name]) for name in choices}
        rows={}
        for category in meta['categories']:
            selected=cats==CATEGORIES.index(category)
            if not selected.any():continue
            ids=np.unique(docs[selected])
            assert len(ids)*meta['length']==int(selected.sum())
            per_doc={name:np.array([scores[name][docs==doc].mean() for doc in ids])
                     for name in choices}
            rows[category]={
                'n_documents':len(ids),
                'mean_effective_coordinates':{name:float(per_doc[name].mean())
                                              for name in choices},
                'balanced_minus_weight':bootstrap_difference(
                    per_doc['balanced_fit_categories'],per_doc['weight_0'],rng),
                'balanced_minus_pooled_oracle':bootstrap_difference(
                    per_doc['balanced_fit_categories'],
                    per_doc['pooled_all_categories_oracle'],rng),
            }
        output[panel]=rows
    a=rotations['balanced_fit_categories']
    stability={}
    for name in ('pooled_all_categories_oracle','held_categories_oracle'):
        overlap=np.abs(a.T@rotations[name])
        i,j=linear_sum_assignment(-overlap)
        stability[name]={'matched_axis_cosine_mean':float(overlap[i,j].mean()),
                         'matched_axis_cosine_median':float(np.median(overlap[i,j]))}
    result={'stem':stem,'scope':'paired document bootstrap, fixed rotations, no refit',
            'panels':output,'axis_overlap':stability,
            'fit_category_axis_transfer':{
                mode:category_axis_transfer(arrays,meta,
                    rotations['balanced_fit_categories'],mode)
                for mode in ('all_key_positions','fixed_key_position')},
            'caveat':'documents are sampled conditionally on next-token form; gradient sparsity does not imply a stable semantic component or finite causal effect'}
    (ART/f'{stem}_document_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    run()

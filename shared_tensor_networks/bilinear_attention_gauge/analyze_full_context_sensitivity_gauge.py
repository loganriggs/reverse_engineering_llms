"""Document-level uncertainty and independent-start stability for QK1 gauges."""

import argparse
import json

import numpy as np
from scipy.optimize import linear_sum_assignment

from flagship_shared_key_gauge_priors import rotation,varimax
from flagship_transfer import ART


LAWS=('test_fineweb','test_wikitext103','test_tinystories')
METHODS=('native','random_0','weight_0','response_0','full_context_varimax')


def per_document_effective(x,r,length):
    y=(x.astype(np.float64)@r).reshape(-1,length,r.shape[1])
    power=np.sum(y*y,axis=-1)
    return np.mean(np.sum(np.abs(y),axis=-1)**2/
                   np.maximum(power,1e-30),axis=1)


def axis_match(a,b):
    similarity=np.abs(a.T@b)
    i,j=linear_sum_assignment(-similarity)
    return dict(mean_abs_cosine=float(np.mean(similarity[i,j])),
                median_abs_cosine=float(np.median(similarity[i,j])),
                fraction_above_0_8=float(np.mean(similarity[i,j]>.8)))


def run(stem,bootstrap=10000,extra_starts=2):
    metadata=json.loads((ART/f'{stem}.json').read_text())
    arrays=np.load(ART/f'{stem}.npz')
    length=metadata['length']
    original=arrays['rotation_full_context_varimax']
    train=arrays['train_fineweb_vectors'].astype(np.float64)
    norms=np.linalg.norm(train,axis=1)
    normalized=train/np.maximum(norms[:,None],metadata['train_normalization_floor'])
    starts={'full_context_varimax':original}
    for seed in range(extra_starts):
        fitted,_=varimax(normalized,rotation(21000+seed,256),max_iter=240)
        starts[f'full_context_random_start_{seed}']=fitted
    rng=np.random.default_rng(82137)
    corpora={}
    for law in LAWS:
        x=arrays[f'{law}_vectors']
        per={method:per_document_effective(x,arrays[f'rotation_{method}'],length)
             for method in METHODS}
        for name,r in starts.items():
            if name not in per:per[name]=per_document_effective(x,r,length)
        idx=rng.integers(0,len(per['native']),size=(bootstrap,len(per['native'])))
        paired={}
        for challenger in ('native','random_0','weight_0','response_0'):
            diff=per['full_context_varimax']-per[challenger]
            dist=diff[idx].mean(1)
            paired[challenger]=dict(mean_difference=float(np.mean(diff)),
                bootstrap_95_percentile=list(map(float,np.quantile(dist,[.025,.975]))),
                improved_documents=int(np.sum(diff<0)))
        energy=np.sum(x.astype(np.float64)**2,axis=1).reshape(-1,length).sum(1)
        corpora[law]=dict(n_documents=len(per['native']),
            gradient_energy_effective_documents=float(energy.sum()**2/
                                                       np.sum(energy**2)),
            max_document_energy_share=float(energy.max()/energy.sum()),
            effective_coordinates={name:float(np.mean(values))
                                   for name,values in per.items()},
            paired=paired)
    matches={name:axis_match(original,r) for name,r in starts.items()
             if name!='full_context_varimax'}
    if extra_starts>=2:
        matches['random_start_0_vs_1']=axis_match(
            starts['full_context_random_start_0'],starts['full_context_random_start_1'])
    output=dict(source_stem=stem,bootstrap_unit='document',
        bootstrap_draws=bootstrap,extra_starts=extra_starts,
        fit_training_rows=len(train),metric='mean per-key-position effective coordinate count',
        methods=list(METHODS),corpora=corpora,axis_matches=matches,
        caveat='sensitivity sparsity is not semantic identification; energy-weighted label enrichment is tail-sensitive')
    (ART/f'{stem}_analysis.json').write_text(json.dumps(output,indent=2)+'\n')
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('stem')
    p.add_argument('--bootstrap',type=int,default=10000)
    p.add_argument('--extra-starts',type=int,default=2)
    a=p.parse_args()
    run(a.stem,a.bootstrap,a.extra_starts)

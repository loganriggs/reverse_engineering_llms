"""Prompt-level transfer and frozen axis/position selection for task gauges."""

import json

import numpy as np

from flagship_explicit_task_gauge import FAMILIES, FIT_FAMILIES, STEM, effective_axes
from flagship_transfer import ART


def energy(arrays,rotation,panel):
    x=arrays[f'{panel}_gradients'].astype(np.float64)@rotation
    e=np.sum(x*x,axis=1)
    return e/np.maximum(e.sum(1,keepdims=True),1e-30)


def select_axes(arrays,rotation,families):
    train=energy(arrays,rotation,'train')
    labels=arrays['train_family']
    gradients=arrays['train_gradients'].astype(np.float64)@rotation
    lengths=arrays['train_lengths']
    result={}
    for family in families:
        cid=FAMILIES.index(family)
        others=np.isin(labels,[FAMILIES.index(x) for x in families if x!=family])
        numerator=train[labels==cid].mean(0)
        denominator=train[others].mean(0)
        axis=int(np.argmax((numerator+1e-5)/(denominator+1e-5)))
        positional=np.zeros(6)
        for i in np.where(labels==cid)[0]:
            length=int(lengths[i]);g=gradients[i,:length,axis]**2
            g/=max(g.sum(),1e-30)
            for offset in range(min(length,6)):
                positional[offset]+=g[length-1-offset]
        positional/=int(np.sum(labels==cid))
        offset=int(np.argmax(positional))
        transfer={}
        for panel in ('train','validation','test'):
            e=energy(arrays,rotation,panel)
            f=arrays[f'{panel}_family']
            other=np.isin(f,[FAMILIES.index(x) for x in families if x!=family])
            transfer[panel]=float((e[f==cid,axis].mean()+1e-5)/
                                  (e[other,axis].mean()+1e-5))
        result[family]={'axis':axis,'offset_from_end':offset,
                        'train_position_energy_fractions':positional.tolist(),
                        'energy_enrichment':transfer}
    return result


def run(stem=STEM):
    meta=json.loads((ART/f'{stem}.json').read_text())
    arrays=np.load(ART/f'{stem}.npz')
    rotations=np.load(ART/f'{stem}_rotations.npz')
    selections={
        'explicit_fit_families':select_axes(arrays,rotations['explicit_fit_families'],FIT_FAMILIES),
        'all_families_oracle':select_axes(arrays,rotations['all_families_oracle'],FAMILIES),
    }
    rng=np.random.default_rng(10411)
    paired={}
    for panel in ('validation','test'):
        paired[panel]={}
        labels=arrays[f'{panel}_family']
        gradients=arrays[f'{panel}_gradients']
        lengths=arrays[f'{panel}_lengths']
        values={name:np.array([effective_axes(gradients[i],int(lengths[i]),rotations[name])
                               for i in range(len(labels))])
                for name in ('corpus_full_context','explicit_fit_families',
                             'fit_plus_antonym','fit_plus_country_capital',
                             'all_families_oracle')}
        for family in FAMILIES:
            selected=labels==FAMILIES.index(family)
            item={}
            for a,b in (('explicit_fit_families','corpus_full_context'),
                        ('all_families_oracle','explicit_fit_families')):
                diff=values[a][selected]-values[b][selected]
                indices=rng.integers(0,len(diff),(10000,len(diff)))
                item[f'{a}_minus_{b}']={
                    'mean':float(diff.mean()),
                    'ci95':[float(v) for v in np.quantile(diff[indices].mean(1),[.025,.975])],
                    'n_prompts':int(selected.sum())}
            paired[panel][family]=item
    result={'stem':stem,'axis_selection':selections,
            'paired_effective_axis_differences':paired,
            'caveat':'prompt variants reuse finite templates and lexical items; bootstrap intervals do not establish new-template generalization'}
    (ART/f'{stem}_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    run()

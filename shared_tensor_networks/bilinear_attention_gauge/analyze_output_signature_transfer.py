"""Assess frozen signed output direction and category transfer by document."""

import json

import numpy as np

from flagship_output_signature_holdout import DISCOVERY
from flagship_transfer import ART


def cosine(a,b):
    return float(a@b/max(np.linalg.norm(a)*np.linalg.norm(b),1e-30))


def run(stem=DISCOVERY,bootstrap=10000):
    discovery=json.loads((ART/f'{stem}.json').read_text())
    arrays=np.load(ART/f'{stem}.npz')
    holdout=json.loads((ART/f'{stem}_holdout.json').read_text())
    methods=tuple(discovery['methods'])
    split={}
    for method in methods:
        x=arrays[f'{method}_delta'].astype(np.float64)
        d=x[:,0]-x[:,1]
        split[method]=dict(
            first_half_cosine=cosine(d[:len(d)//2].mean(0),d[len(d)//2:].mean(0)),
            even_odd_cosine=cosine(d[::2].mean(0),d[1::2].mean(0)))
    rng=np.random.default_rng(19031)
    corpora={}
    for corpus in ('fineweb_test','wikitext103','tinystories'):
        corpora[corpus]={}
        for method in methods:
            rows=[r for r in holdout['records']
                  if r['corpus']==corpus and r['method']==method]
            assert len(rows)==12
            cos=np.array([r['cosine_to_discovery_mean'] for r in rows])
            contrast=np.array([r['output_category_contrast'] for r in rows])
            idx=rng.integers(0,len(rows),size=(bootstrap,len(rows)))
            corpora[corpus][method]=dict(
                n_documents=len(rows),
                mean_cosine=float(cos.mean()),
                cosine_bootstrap_95_percentile=list(map(float,np.quantile(
                    cos[idx].mean(1),[.025,.975]))),
                positive_cosine_documents=int(np.sum(cos>0)),
                mean_category_contrast=float(contrast.mean()),
                category_contrast_bootstrap_95_percentile=list(map(float,np.quantile(
                    contrast[idx].mean(1),[.025,.975]))),
                category_sign_matching_documents=int(sum(
                    r['output_category_contrast_sign_matches'] for r in rows)))
    output=dict(source_discovery=stem,source_holdout=stem+'_holdout',
        bootstrap_unit='document',bootstrap=bootstrap,
        split_half_discovery=split,corpora=corpora,
        verdict_scope='frozen signed output signature and category contrast, not deletion energy or native circuit identity')
    (ART/f'{stem}_transfer_analysis.json').write_text(json.dumps(output,indent=2)+'\n')
    return output


if __name__=='__main__':run()

"""Coarse held-panel form check for the nested-span toy language."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_ROWS, DEFAULT_TOKENIZER
from surface_mixture_v2_nested import SurfaceMixtureV2Nested, render_nested
from surface_mixture_v2_nested_varlen import SurfaceMixtureV2NestedVarlen
from surface_mixture_v3_nested_varlen import SurfaceMixtureV3NestedVarlen


def counts(texts):
    words = [re.findall(r"[A-Za-z]+", text) for text in texts]
    return dict(n=len(texts), distinct_words=len(set(w.lower() for row in words
        for w in row)), mean_words=sum(map(len, words))/len(words),
        parenthesis_window_fraction=sum('(' in text and ')' in text
                                         for text in texts)/len(texts),
        square_window_fraction=sum('[' in text and ']' in text
                                    for text in texts)/len(texts),
        newline_window_fraction=sum('\n' in text for text in texts)/len(texts),
        mean_openers=sum(text.count('(')+text.count('[')
                         for text in texts)/len(texts))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rows',type=Path,default=DEFAULT_ROWS)
    p.add_argument('--tokenizer',type=Path,default=DEFAULT_TOKENIZER)
    p.add_argument('--n',type=int,default=480)
    p.add_argument('--length',type=int,default=128)
    p.add_argument('--seed',type=int,default=20261205)
    p.add_argument('--generator',choices=('fixed','varlen','joint'),default='fixed')
    p.add_argument('--object-vocab-size',type=int,default=8)
    p.add_argument('--nested-mass',type=float,default=.3)
    p.add_argument('--inner-mass',type=float,default=.6)
    p.add_argument('--extra-mass',type=float,default=.5)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    rows=torch.load(a.rows,map_location='cpu',weights_only=True)
    assert rows.shape[0]>=a.n and rows.shape[1]>a.length
    tokenizer=Tokenizer.from_file(str(a.tokenizer))
    natural=[tokenizer.decode(row[:a.length].tolist(),skip_special_tokens=False)
             for row in rows[:a.n]]
    cls={'fixed':SurfaceMixtureV2Nested,
         'varlen':SurfaceMixtureV2NestedVarlen,
         'joint':SurfaceMixtureV3NestedVarlen}[a.generator]
    kwargs=dict(seed=a.seed,nested_mass=a.nested_mass,
                inner_mass=a.inner_mass)
    if a.generator in ('varlen','joint'):
        kwargs['extra_mass']=a.extra_mass
    if a.generator=='joint':
        kwargs['object_vocab_size']=a.object_vocab_size
    gen=cls(**kwargs)
    synthetic=[render_nested(gen.sample_sequence(a.length)[0])
               for _ in range(a.n)]
    result=dict(config={**vars(a),'rows':str(a.rows),
                        'tokenizer':str(a.tokenizer),'output':str(a.output)},
                source_sha256=hashlib.sha256(rows.numpy().tobytes()).hexdigest(),
                natural=counts(natural),synthetic=counts(synthetic),
                caveat='Same count of tokens but different tokenizers and '
                       'non-document windows. Character-form frequencies are '
                       'not semantic task prevalence. The nested syntax is '
                       'deliberately artificial and these numbers are a '
                       'negative realism check.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print('natural',result['natural'])
    print('synthetic',result['synthetic'])


if __name__=='__main__':
    main()

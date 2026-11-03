"""New, disjoint FineWeb form screen for V12; exports aggregate statistics only."""

import argparse
import hashlib
import json
import re
from itertools import islice
from pathlib import Path

from datasets import load_dataset

from audit_surface_repetition import form_cues, score
from surface_mixture_v6 import render
from surface_mixture_v12_price_faq import SurfaceMixtureV12PriceFAQ
from train_v12_price_faq import COMMON, REFERENCE


WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def first_words(text, n=100):
    matches=list(WORD.finditer(text))
    return text[:matches[n-1].end()] if len(matches)>=n else text


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--skip',type=int,default=15000)
    p.add_argument('--n',type=int,default=128)
    p.add_argument('--words',type=int,default=100)
    p.add_argument('--toy-length',type=int,default=256)
    p.add_argument('--seed',type=int,default=99801)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    dataset=load_dataset('HuggingFaceFW/fineweb',name='sample-10BT',
                         split='train',streaming=True)
    selected=list(islice(dataset,a.skip,a.skip+a.n))
    assert len(selected)==a.n
    texts=[row['text'] for row in selected]
    hashes=[hashlib.sha256(text.encode()).hexdigest() for text in texts]
    gen=SurfaceMixtureV12PriceFAQ(seed=a.seed,**COMMON,**REFERENCE)
    synthetic=[render([gen.id2token[i] for i in gen.sample_sequence(
        a.toy_length)[0]]) for _ in selected]
    natural=[first_words(v,a.words) for v in texts]
    toy=[first_words(v,a.words) for v in synthetic]
    eligible=[i for i in range(a.n) if len(WORD.findall(natural[i]))>=a.words
              and len(WORD.findall(toy[i]))>=a.words]
    natural=[natural[i] for i in eligible]
    toy=[toy[i] for i in eligible]
    assert natural and toy
    result=dict(config={**vars(a),'output':str(a.output),
                        'dataset':'HuggingFaceFW/fineweb sample-10BT train',
                        'selection':'contiguous streaming rows, prespecified '
                                    'offset, then >=100-word matched eligibility'},
        selected_text_sha256=hashes,n_word_matched=len(eligible),
        natural=dict(form=form_cues(natural),repetition=score(natural,40)),
        v12_reference_candidate=dict(form=form_cues(toy),
                                     repetition=score(toy,40)),
        caveat='Fresh offset for this toy-language line, disjoint from local '
               'skip-7000 and skip-11000 development panels. Aggregate form '
               'check only: different tokenizers and text lengths, no semantic '
               'dependency annotations or proof of naturalness. No source prose '
               'is exported.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print('word matched',len(eligible),'FineWeb',result['natural'],
          'V12',result['v12_reference_candidate'],flush=True)


if __name__=='__main__':main()

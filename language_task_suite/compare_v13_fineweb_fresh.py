"""Prespecified fresh FineWeb form screen for the V13 attribute stream."""

import argparse
import hashlib
import json
from itertools import islice
from pathlib import Path

from datasets import load_dataset

from audit_surface_repetition import form_cues, score
from compare_v12_fineweb_fresh import WORD, first_words
from surface_mixture_v6 import render
from surface_mixture_v13_attributes import SurfaceMixtureV13Attributes
from train_v13_attributes import COMMON, REFERENCE


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--skip',type=int,default=16000)
    p.add_argument('--n',type=int,default=64)
    p.add_argument('--words',type=int,default=100)
    p.add_argument('--toy-length',type=int,default=256)
    p.add_argument('--seed',type=int,default=99821)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    ds=load_dataset('HuggingFaceFW/fineweb',name='sample-10BT',
                    split='train',streaming=True)
    selected=list(islice(ds,args.skip,args.skip+args.n))
    assert len(selected)==args.n
    texts=[row['text'] for row in selected]
    hashes=[hashlib.sha256(v.encode()).hexdigest() for v in texts]
    gen=SurfaceMixtureV13Attributes(seed=args.seed,**COMMON,**REFERENCE)
    synthetic=[render([gen.id2token[i] for i in gen.sample_sequence(
        args.toy_length)[0]]) for _ in selected]
    natural=[first_words(v,args.words) for v in texts]
    toy=[first_words(v,args.words) for v in synthetic]
    eligible=[i for i in range(args.n)
              if len(WORD.findall(natural[i]))>=args.words
              and len(WORD.findall(toy[i]))>=args.words]
    assert eligible
    natural=[natural[i] for i in eligible]
    toy=[toy[i] for i in eligible]
    result=dict(config={**vars(args),'output':str(args.output),
                        'dataset':'HuggingFaceFW/fineweb sample-10BT train',
                        'selection':'contiguous prespecified streaming rows, '
                                    'then word-matched eligibility'},
        selected_text_sha256=hashes,n_word_matched=len(eligible),
        natural=dict(form=form_cues(natural),repetition=score(natural,40)),
        v13_reference_candidate=dict(form=form_cues(toy),
                                     repetition=score(toy,40)),
        caveat='Fresh offset, disjoint from earlier FineWeb screens. Form '
               'statistics do not establish semantic realism. Different '
               'tokenizers and document lengths. No source prose is exported.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('matched',len(eligible),'natural',result['natural'],
          'toy',result['v13_reference_candidate'],flush=True)


if __name__=='__main__':main()

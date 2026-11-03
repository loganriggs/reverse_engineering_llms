"""Fresh, text-free aggregate form screen for V14 versus V16 and FineWeb."""

import argparse
import gc
import hashlib
import json
import os
import sys
from itertools import islice
from pathlib import Path

from datasets import load_dataset

from audit_surface_repetition import WORD, form_cues, score
from audit_v14_repetition_attribution import attribution
from compare_v12_fineweb_fresh import first_words
from surface_mixture_v6 import render
from surface_mixture_v14_conditional import SurfaceMixtureV14Conditional
from surface_mixture_v16_determiners import SurfaceMixtureV16Determiners
from train_v13_attributes import COMMON, REFERENCE


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--skip',type=int,default=26000)
    p.add_argument('--n',type=int,default=64)
    p.add_argument('--words',type=int,default=100)
    p.add_argument('--prefix-words',type=int,default=40)
    p.add_argument('--toy-length',type=int,default=256)
    p.add_argument('--seed',type=int,default=100301)
    p.add_argument('--article-variation',type=float,default=.8)
    p.add_argument('--indefinite-share',type=float,default=.9)
    p.add_argument('--fact-article-variation',type=float,default=.2)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    ds=load_dataset('HuggingFaceFW/fineweb',name='sample-10BT',
                    split='train',streaming=True)
    stream=iter(ds)
    try:
        selected=list(islice(stream,args.skip,args.skip+args.n))
    finally:
        if hasattr(stream,'close'):
            stream.close()
    del stream, ds
    gc.collect()
    assert len(selected)==args.n
    natural=[row['text'] for row in selected]
    hashes=[hashlib.sha256(text.encode()).hexdigest() for text in natural]
    kwargs=dict(COMMON,**REFERENCE,conditional_continuation_rate=.35)
    v16_kwargs=dict(article_variation=args.article_variation,
                    indefinite_share=args.indefinite_share,
                    fact_article_variation=args.fact_article_variation)
    texts={}
    for name,cls in (('v14',SurfaceMixtureV14Conditional),
                     ('v16',SurfaceMixtureV16Determiners)):
        gen=cls(seed=args.seed,**kwargs,
                **(v16_kwargs if name=='v16' else {}))
        texts[name]=[first_words(render([gen.id2token[i] for i in
                    gen.sample_sequence(args.toy_length)[0]]),args.words)
                    for _ in selected]
    texts['natural']=[first_words(text,args.words) for text in natural]
    eligible=[i for i in range(args.n) if all(
        len(WORD.findall(values[i]))>=args.words
        for values in texts.values())]
    assert eligible
    panels={}
    for name,values in texts.items():
        cropped=[values[i] for i in eligible]
        panels[name]=dict(form=form_cues(cropped),
                          repetition=score(cropped,args.prefix_words),
                          attribution=attribution(cropped,args.prefix_words))
    result=dict(config={**vars(args),'output':str(args.output),
                        'generator_kwargs':kwargs,
                        'v16_kwargs':v16_kwargs,
                        'dataset':'HuggingFaceFW/fineweb sample-10BT train',
                        'selection':'contiguous streaming rows then 100-word '
                                    'eligibility in all arms'},
                n_word_matched=len(eligible),selected_text_sha256=hashes,
                panels=panels,
                caveat='Small diagnostic sample and different tokenizers; '
                       'no natural text or toy samples are saved. Form '
                       'similarity is not semantic realism or task learning.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    for name,row in panels.items():
        print(name,'repeat',row['repetition']['repeated_word_fraction'],
              'the',dict(row['attribution']['top_word_occurrences']).get('the',0),
              'stopword repeats',row['attribution']['repeated_stopword_occurrences'],
              flush=True)
    # Deep FineWeb streaming leaves a background worker waiting at Python
    # shutdown in this environment. The JSON file has been closed and stdout
    # flushed, so terminate only after the full comparison succeeds.
    sys.stdout.flush()
    os._exit(0)


if __name__=='__main__':main()

"""Aggregate form comparison for fixed and variable world-document renderers."""

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
from document_world_v4_interleaved import sample_documents as sample_v4
from document_world_v5_clause_realizations import sample_documents as sample_v5


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--skip',type=int,default=28000)
    p.add_argument('--n',type=int,default=64)
    p.add_argument('--words',type=int,default=100)
    p.add_argument('--prefix-words',type=int,default=40)
    p.add_argument('--seed',type=int,default=33071)
    p.add_argument('--style-entropy',type=float,default=.7)
    p.add_argument('--aside-rate',type=float,default=.3)
    p.add_argument('--paragraph-rate',type=float,default=.22)
    p.add_argument('--header-rate',type=float,default=.5)
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
    del stream,ds
    gc.collect()
    assert len(selected)==args.n
    natural=[row['text'] for row in selected]
    hashes=[hashlib.sha256(text.encode()).hexdigest() for text in natural]
    v4=sample_v4(args.n,args.seed)
    v5=sample_v5(args.n,args.seed,style_entropy=args.style_entropy,
                 aside_rate=args.aside_rate,paragraph_rate=args.paragraph_rate,
                 header_rate=args.header_rate)
    texts=dict(v4=[first_words(d.text(),args.words) for d in v4],
               v5=[first_words(d.text(),args.words) for d in v5],
               natural=[first_words(s,args.words) for s in natural])
    eligible=[i for i in range(args.n) if all(
        len(WORD.findall(values[i]))>=args.words
        for values in texts.values())]
    assert eligible
    panels={}
    for name,values in texts.items():
        cropped=[values[i] for i in eligible]
        words=[w.lower() for text in cropped for w in WORD.findall(text)]
        from collections import Counter
        counts=Counter(words)
        panels[name]=dict(form=form_cues(cropped),
                          repetition=score(cropped,args.prefix_words),
                          attribution=attribution(cropped,args.prefix_words),
                          lexical=dict(types=len(counts),
                                       top10_share=sum(v for _,v in
                                                       counts.most_common(10))
                                       /len(words)))
    out=dict(config={**vars(args),'output':str(args.output),
                     'dataset':'HuggingFaceFW/fineweb sample-10BT train',
                     'selection':'contiguous streaming rows, then jointly '
                                 '100-word-eligible first-window comparison'},
             n_word_matched=len(eligible),selected_text_sha256=hashes,
             panels=panels,
             caveat='Small form diagnostic with differing tokenization; '
                    'no natural text or generated documents saved. '
                    'This does not establish semantic realism.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,indent=2)+'\n')
    for name,row in panels.items():
        print(name,'types',row['lexical']['types'],
              'top10',row['lexical']['top10_share'],
              'repeat',row['repetition']['repeated_word_fraction'],
              flush=True)
    # Deep streaming can leave a background worker waiting at interpreter
    # shutdown here. All output is closed and flushed before this exit.
    sys.stdout.flush()
    os._exit(0)


if __name__=='__main__':main()

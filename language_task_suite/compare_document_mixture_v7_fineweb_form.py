"""Long-document form screen for the V6 and V7 task mixtures."""

import argparse
import gc
import hashlib
import json
import os
import random
import sys
from collections import Counter
from itertools import islice
from pathlib import Path

from datasets import load_dataset

from audit_surface_repetition import WORD,form_cues,score
from compare_v12_fineweb_fresh import first_words
from document_mixture_v6 import sample_document as sample_v6
from document_mixture_v7_cross import sample_document as sample_v7


def generated(n,seed,version,words):
    rng=random.Random(seed)
    texts=[];kinds=Counter();draws=0
    while len(texts)<n:
        draws+=1
        if version=='v6':
            kind=rng.choice(('narrative','event_report'))
            doc=sample_v6(rng,kind=kind)
        else:
            kind=rng.choices(('narrative','event_report','cross_report'),
                             (.375,.375,.25))[0]
            doc=sample_v7(rng,kind=kind)
        cropped=first_words(doc.text(),words)
        if len(WORD.findall(cropped))>=words:
            texts.append(cropped)
            kinds[kind]+=1
    return texts,dict(kinds),draws


def panel(texts,prefix_words):
    words=[word.lower() for text in texts for word in WORD.findall(text)]
    freq=Counter(words)
    return dict(n=len(texts),form=form_cues(texts),
                repetition=score(texts,prefix_words),
                word_types=len(freq),
                hapax_types=sum(value==1 for value in freq.values()),
                top10_share=sum(v for _,v in freq.most_common(10))/len(words))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--skip',type=int,default=30000)
    p.add_argument('--n',type=int,default=64)
    p.add_argument('--words',type=int,default=100)
    p.add_argument('--prefix-words',type=int,default=40)
    p.add_argument('--seed',type=int,default=34781)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    ds=load_dataset('HuggingFaceFW/fineweb',name='sample-10BT',
                    split='train',streaming=True)
    stream=iter(ds)
    try:
        selected=list(islice(stream,a.skip,a.skip+a.n*3))
    finally:
        if hasattr(stream,'close'):
            stream.close()
    del stream,ds
    gc.collect()
    natural=[];hashes=[]
    for row in selected:
        cropped=first_words(row['text'],a.words)
        if len(WORD.findall(cropped))>=a.words:
            natural.append(cropped)
            hashes.append(hashlib.sha256(row['text'].encode()).hexdigest())
        if len(natural)==a.n:
            break
    assert len(natural)==a.n
    v6,kinds6,draws6=generated(a.n,a.seed,'v6',a.words)
    v7,kinds7,draws7=generated(a.n,a.seed+1,'v7',a.words)
    panels={name:panel(texts,a.prefix_words) for name,texts in
            (('v6',v6),('v7',v7),('fineweb',natural))}
    out=dict(config={**vars(a),'output':str(a.output),
                     'dataset':'HuggingFaceFW/fineweb sample-10BT train',
                     'selection':'first n >=100-word windows from n*3 '
                                 'contiguous rows; long generated document '
                                 'kinds sampled separately until eligible'},
             generated_long_kind_counts={'v6':kinds6,'v7':kinds7},
             generated_draws={'v6':draws6,'v7':draws7},
             selected_text_sha256=hashes,panels=panels,
             caveat='Long-document conditional form screen, not overall '
                    'mixture prevalence or semantic realism. No source text '
                    'or generated document bodies are saved.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,indent=2)+'\n')
    for name,row in panels.items():
        print(name,row['word_types'],round(row['top10_share'],3),
              round(row['repetition']['repeated_word_fraction'],3),flush=True)
    sys.stdout.flush()
    # This dataset stream can leave a worker waiting at interpreter shutdown.
    os._exit(0)


if __name__=='__main__':main()

"""Fresh FineWeb panel: attribute repeated words to lexical classes.

Only hashes and aggregate counts are saved; source documents are not copied.
"""

import argparse
import hashlib
import json
from collections import Counter
from itertools import islice
from pathlib import Path

from datasets import load_dataset

from audit_surface_repetition import STOPWORDS, WORD, form_cues, score
from compare_v12_fineweb_fresh import first_words
from surface_mixture_v6 import render
from surface_mixture_v14_conditional import SurfaceMixtureV14Conditional
from surface_mixture_v15_tail_variants import SurfaceMixtureV15TailVariants
from train_v13_attributes import COMMON, REFERENCE


def attribution(texts,prefix_words):
    repeated=Counter();first_counts=Counter();stop_repeat=0;content_repeat=0
    total_words=0;stop_words=0
    for text in texts:
        words=[w.lower() for w in WORD.findall(text)[:prefix_words]]
        seen=set()
        for word in words:
            first_counts[word]+=1
            total_words+=1
            if word in STOPWORDS:stop_words+=1
            if word in seen:
                repeated[word]+=1
                if word in STOPWORDS:stop_repeat+=1
                else:content_repeat+=1
            seen.add(word)
    return dict(total_words=total_words,stopword_occurrences=stop_words,
                repeated_stopword_occurrences=stop_repeat,
                repeated_other_occurrences=content_repeat,
                top_repeated_words=repeated.most_common(25),
                top_word_occurrences=first_counts.most_common(25))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--skip',type=int,default=19000)
    p.add_argument('--n',type=int,default=64)
    p.add_argument('--words',type=int,default=100)
    p.add_argument('--prefix-words',type=int,default=40)
    p.add_argument('--toy-length',type=int,default=256)
    p.add_argument('--seed',type=int,default=99841)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    ds=load_dataset('HuggingFaceFW/fineweb',name='sample-10BT',
                    split='train',streaming=True)
    selected=list(islice(ds,args.skip,args.skip+args.n))
    assert len(selected)==args.n
    natural_raw=[row['text'] for row in selected]
    hashes=[hashlib.sha256(text.encode()).hexdigest() for text in natural_raw]
    kwargs=dict(COMMON,**REFERENCE,conditional_continuation_rate=.35)
    texts={}
    for name,cls in [('v14',SurfaceMixtureV14Conditional),
                     ('v15_pilot',SurfaceMixtureV15TailVariants)]:
        gen=cls(seed=args.seed,**kwargs)
        texts[name]=[first_words(render([gen.id2token[i] for i in
                    gen.sample_sequence(args.toy_length)[0]]),args.words)
                    for _ in selected]
    texts['natural']=[first_words(row,args.words) for row in natural_raw]
    eligible=[i for i in range(args.n) if all(
        len(WORD.findall(values[i]))>=args.words for values in texts.values())]
    assert eligible
    panels={}
    for name,values in texts.items():
        cropped=[values[i] for i in eligible]
        panels[name]=dict(form=form_cues(cropped),
                          repetition=score(cropped,args.prefix_words),
                          attribution=attribution(cropped,args.prefix_words))
    result=dict(config={**vars(args),'output':str(args.output),
                        'dataset':'HuggingFaceFW/fineweb sample-10BT train',
                        'selection':'contiguous streaming rows then 100-word '
                                    'eligibility in all three arms'},
                n_word_matched=len(eligible),selected_text_sha256=hashes,
                panels=panels,
                caveat='A fixed heuristic stopword list separates classes. '
                       'This is lexical-form evidence, not a semantic task '
                       'annotation; dataset and toy tokenization differ. '
                       'The V15 tail variant is an exploratory rejected pilot.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    for name,panel in panels.items():
        print(name,'repeat',panel['repetition']['repeated_word_fraction'],
              'stopword repeats',panel['attribution']['repeated_stopword_occurrences'],
              'other repeats',panel['attribution']['repeated_other_occurrences'],
              'top',panel['attribution']['top_repeated_words'][:8],flush=True)


if __name__=='__main__':main()

"""Exact-prefix and task-coverage audit for the V15 surface pilot."""

import argparse
import json
from collections import Counter
from pathlib import Path

from audit_surface_repetition import form_cues, score
from audit_v13_attributes import panel as inherited_panel
from audit_v14_conditional import conditional_panel, repair_completed_conditional
from compare_v12_fineweb_fresh import first_words
from surface_mixture_v6 import render
from surface_mixture_v14_conditional import SurfaceMixtureV14Conditional
from surface_mixture_v15_tail_variants import (
    TAIL_BRANCH, SurfaceMixtureV15TailVariants,
)
from train_v13_attributes import COMMON, REFERENCE


def surface_panel(cls,seed,n,length,words,repetition_words,kwargs):
    gen=cls(seed=seed,**kwargs)
    texts=[];labels=Counter();branches=Counter()
    for _ in range(n):
        ids,labs,_=gen.sample_sequence(length)
        tokens=[gen.id2token[i] for i in ids]
        labels.update(labs)
        for i in range(len(tokens)-2):
            if tokens[i]=='because' and tokens[i+2] in TAIL_BRANCH:
                branches[tokens[i+2]]+=1
        texts.append(first_words(render(tokens),words))
    expected=(TAIL_BRANCH if cls is SurfaceMixtureV15TailVariants
              else {'had':1.})
    assert all(branches[branch]>0 for branch in expected)
    return dict(generator=cls.__name__,seed=seed,n=n,length=length,
                labels=dict(labels),tail_branches=dict(branches),
                form=form_cues(texts),repetition=score(texts,repetition_words))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,default=99831)
    p.add_argument('--n',type=int,default=100)
    p.add_argument('--surface-n',type=int,default=256)
    p.add_argument('--length',type=int,default=256)
    p.add_argument('--words',type=int,default=100)
    p.add_argument('--repetition-words',type=int,default=40)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    kwargs=dict(COMMON,**REFERENCE,conditional_continuation_rate=.35)
    exact=[];coverage=[]
    for seed in (args.seed,args.seed+1):
        inherited=inherited_panel(seed,args.n,args.length,kwargs,
                                  SurfaceMixtureV15TailVariants,
                                  repair_completed_conditional)
        conditional=conditional_panel(seed,args.n,args.length,kwargs,
                                      SurfaceMixtureV15TailVariants)
        assert inherited['labels']==conditional['labels']
        exact.append(dict(seed=seed,inherited=inherited,conditional=conditional))
        older=surface_panel(SurfaceMixtureV14Conditional,seed,args.surface_n,
                            args.length,args.words,args.repetition_words,kwargs)
        newer=surface_panel(SurfaceMixtureV15TailVariants,seed,args.surface_n,
                            args.length,args.words,args.repetition_words,kwargs)
        coverage.append(dict(seed=seed,v14=older,v15=newer))
        print(seed,'repeat V14/V15',
              older['repetition']['repeated_word_fraction'],
              newer['repetition']['repeated_word_fraction'],
              'conditional sites',older['labels'].get('conditional_action'),
              newer['labels'].get('conditional_action'),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(
        config={**vars(args),'output':str(args.output),
                'generator_kwargs':kwargs},exact_panels=exact,
        surface_panels=coverage,
        caveat='This is a versioned surface-only pilot. Exact prefix and '
               'supported edit checks show DGP correctness, not linguistic '
               'realism. Form statistics use word-matched toy-only windows; '
               'the natural comparison is the separately held-out FineWeb '
               'offset-18000 panel from the V14 screen.'),indent=2)+'\n')


if __name__=='__main__':main()

"""Audit V16 exact support, inherited rules, and article/noun marginal."""

import argparse
import json
from collections import Counter
from pathlib import Path

from audit_v13_attributes import panel as inherited_panel
from audit_v14_conditional import (conditional_panel,
                                   repair_completed_conditional)
from surface_mixture_v2 import _frame, _split
from surface_mixture_v2_varied import DETERMINER_SLOTS
from surface_mixture_v4 import _content
from surface_mixture_v12_price_faq import SPECIAL
from surface_mixture_v16_determiners import (
    SurfaceMixtureV16Determiners, _indefinite_article,
)
from train_v13_attributes import COMMON, REFERENCE


def article_checks(seed, n, length, kwargs):
    gen=SurfaceMixtureV16Determiners(seed=seed,**kwargs)
    counts=Counter(); marginals=0
    for _ in range(n):
        ids,_,qs=gen.sample_sequence(length)
        words=[gen.id2token[i] for i in ids]
        for t,(word,q) in enumerate(zip(words,qs)):
            _,raw=_split(words[:t]);content=_content(raw)
            if not content or content[0] in SPECIAL:
                continue
            try:frame=_frame(content)
            except ValueError:continue
            if len(content) not in DETERMINER_SLOTS[frame]:
                continue
            counts[word]+=1
            assert word in ('the','this','that','a','an')
            if word in ('a','an') and t+1<len(words):
                assert _indefinite_article(words[t+1])==word
            if marginals>=120:
                continue
            prefix=words[:t]
            base=super(SurfaceMixtureV16Determiners,gen).next_distribution(
                prefix+['the'])
            mixed=Counter()
            for article,mass in q.items():
                after=gen.next_distribution(prefix+[article])
                for candidate,p in after.items():
                    mixed[candidate]+=mass*p
            assert set(mixed)==set(base)
            assert max(abs(mixed[w]-p) for w,p in base.items())<1e-10
            marginals+=1
    assert counts['a'] and counts['an'] and counts['this'] and counts['that']
    return dict(article_counts=dict(counts),verified_noun_marginals=marginals)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--n',type=int,default=64)
    p.add_argument('--length',type=int,default=256)
    p.add_argument('--seed',type=int,default=100211)
    p.add_argument('--article-variation',type=float,default=.8)
    p.add_argument('--indefinite-share',type=float,default=.9)
    p.add_argument('--fact-article-variation',type=float,default=.2)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    kwargs=dict(COMMON,**REFERENCE,conditional_continuation_rate=.35,
                article_variation=args.article_variation,
                indefinite_share=args.indefinite_share,
                fact_article_variation=args.fact_article_variation)
    rows=[]
    for seed in (args.seed,args.seed+1):
        inherited=inherited_panel(seed,args.n,args.length,kwargs,
                                  SurfaceMixtureV16Determiners,
                                  repair_completed_conditional)
        conditional=conditional_panel(seed,args.n,args.length,kwargs,
                                      SurfaceMixtureV16Determiners)
        assert inherited['labels']==conditional['labels']
        articles=article_checks(seed+1000,args.n,args.length,kwargs)
        rows.append(dict(seed=seed,inherited=inherited,
                         conditional=conditional,articles=articles))
        print(seed,'articles',articles['article_counts'],
              'action',conditional['metrics']['conditional_action_sites'],
              'attribute',inherited['metrics']['attribute_object_sites'],
              flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(
        config={**vars(args),'output':str(args.output),
                'generator_kwargs':kwargs},panels=rows,
        caveat='Exact sampled-prefix and inherited-rule checks; natural '
               'language form is evaluated separately.'),indent=2)+'\n')


if __name__=='__main__':main()

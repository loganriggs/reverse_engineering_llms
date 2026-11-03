"""Visible-prefix and paired-edit audit for the cross-section document pilot."""

import argparse
import copy
import json
import random
import re
from collections import Counter
from pathlib import Path

from document_mixture_v6 import record_holdout
from document_mixture_v7_cross import (KINDS,draw_cross,realize_cross,
                                       sample_document,site_oracle,
                                       _latest_source)
from document_world import NAMES,replay


WORD=re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def check(doc):
    for site in doc.sites:
        assert doc.tokens[site.index]==site.target
        assert site.target in site.distribution
        distribution=site_oracle(doc,site)
        assert abs(sum(distribution.values())-1)<1e-8
        assert site.target in distribution


def edits(world):
    obj,attr=world['query_object'],world['query_attribute']
    source,j=_latest_source(world['event'],obj,attr)
    mapping=dict(world['assignments'])
    other_source=next(name for name in mapping if name!=source)
    # Change only the relevant report source; both possible reviewers remain.
    actor=replay(world['event']['initial'],world['event']['events'][:j])[
        obj]['owner']
    if other_source!=actor:
        changed=copy.deepcopy(world)
        changed['event']['sources'][j]=other_source
        yield 'relevant_source',changed,True

    other_j=next(k for k in range(len(world['event']['events'])-1,-1,-1)
                 if (world['event']['events'][k]['object'],
                     world['event']['events'][k]['attribute'])!=(obj,attr))
    nuisance=copy.deepcopy(world)
    other_event=world['event']['events'][other_j]
    nuisance_actor=replay(world['event']['initial'],
                          world['event']['events'][:other_j])[
                              other_event['object']]['owner']
    nuisance['event']['sources'][other_j]=next(
        name for name in NAMES if name not in (source,nuisance_actor,
                                                nuisance['event']['sources'][other_j]))
    yield 'irrelevant_source',nuisance,False

    swapped=copy.deepcopy(world)
    a,b=swapped['assignments']
    swapped['assignments']=((a[0],b[1]),(b[0],a[1]))
    yield 'reviewer_ship_swap',swapped,True

    revised=copy.deepcopy(world)
    ship=mapping[source]
    i=revised['record']['ships'].index(ship)
    prior=revised['record']['categories'][i]
    values=list(revised['record']['categories'])
    values[i]=next(value for value in ('11B','9C','7A','6D','4B','3C')
                   if value not in (prior,values[1-i]))
    revised['record']['categories']=tuple(values)
    yield 'target_category',revised,True

    reordered=copy.deepcopy(world)
    reordered['section_order']=tuple(reversed(world['section_order']))
    yield 'section_order',reordered,False


def audit(seed,n_cross,n_mixture):
    rng=random.Random(seed)
    counts=Counter();failures=Counter();lengths=[]
    for split in ('train','composition'):
        for _ in range(n_cross):
            world=draw_cross(rng,split)
            doc=realize_cross(world)
            check(doc)
            site=next(s for s in doc.sites if s.family=='cross_source_record')
            target_ship=dict(world['assignments'])[
                _latest_source(world['event'],world['query_object'],
                               world['query_attribute'])[0]]
            assert target_ship==world['record']['query_ship']
            held=record_holdout(target_ship,site.target)
            assert held==(split=='composition')
            counts[f'{split}:cross']+=1
            for name,altered,should_flip in edits(world):
                other=realize_cross(altered)
                check(other)
                answer=next(s.target for s in other.sites
                            if s.family=='cross_source_record')
                assert (answer!=site.target)==should_flip,(name,seed)
                counts[f'{split}:{name}']+=1
    words=Counter();first_window=0
    for _ in range(n_mixture):
        doc=sample_document(rng)
        check(doc)
        counts[f'kind:{doc.kind}']+=1
        counts.update('site:'+s.family for s in doc.sites)
        lengths.append(len(doc.tokens))
        first=WORD.findall(doc.text())
        if len(first)>=100 and first_window<49:
            words.update(word.lower() for word in first[:100])
            first_window+=1
    assert first_window==49
    return dict(seed=seed,n_cross_per_split=n_cross,n_mixture=n_mixture,
                counts=dict(counts),lengths=dict(min=min(lengths),
                median=sorted(lengths)[len(lengths)//2],max=max(lengths)),
                first49_x100_word_types=len(words),
                first49_x100_top10_share=sum(n for _,n in
                    words.most_common(10))/sum(words.values()),
                caveat='One sampled composition task with visible-only parser; '
                       'the overall mixture remains stylized and does not '
                       'establish real-language form or learned model behavior.')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,default=34701)
    p.add_argument('--cross-per-split',type=int,default=300)
    p.add_argument('--mixture',type=int,default=800)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    rows=[audit(a.seed+i,a.cross_per_split,a.mixture) for i in (0,1)]
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(dict(config={**vars(a),
        'output':str(a.output)},panels=rows),indent=2)+'\n')
    for row in rows:
        print(row['seed'],'kinds',[(k,v) for k,v in row['counts'].items()
                                  if k.startswith('kind:')],
              'types',row['first49_x100_word_types'],flush=True)


if __name__=='__main__':main()

"""Audit V6 visible-prefix sites, paired edits, and structural splits."""

import argparse
import collections
import json
import random
from pathlib import Path

from document_mixture_v5 import KINDS, realize_record, realize_score
from document_mixture_v6 import (record_holdout, sample_document,
                                 score_holdout, site_oracle)


def audit(n_per_kind, seed):
    result = {}
    exact = {}
    for split in ('train', 'composition'):
        rng = random.Random(seed if split == 'train' else seed+1)
        families = collections.Counter()
        shortcuts = collections.Counter()
        edits = collections.Counter()
        for kind in KINDS:
            for _ in range(n_per_kind):
                doc = sample_document(rng, kind, split)
                if kind in ('record_listing', 'score_report'):
                    exact.setdefault((split, kind), set()).add(doc.tokens)
                for site in doc.sites:
                    q = site_oracle(doc, site)
                    assert site.target == doc.tokens[site.index]
                    assert site.target in q and abs(sum(q.values())-1)<1e-8
                    assert set(q)==set(site.distribution)
                    assert max(abs(q[t]-site.distribution[t]) for t in q)<1e-9
                    families[site.family] += 1
                if kind == 'record_listing':
                    w=doc.latent
                    held=[record_holdout(s,c) for s,c in zip(
                        w['ships'],w['categories'],strict=True)]
                    queried=w['ships'].index(w['query_ship'])
                    assert held[queried] if split=='composition' else not any(held)
                    shortcuts['record_query_last' if w['ships'][w['order'][-1]]
                              ==w['query_ship'] else 'record_query_nonlast'] +=1
                    new=dict(w)
                    new['query_ship']=w['ships'][1-queried]
                    changed=realize_record(new)
                    assert changed.sites[0].target != doc.sites[0].target
                    assert site_oracle(changed,changed.sites[0]) == {
                        changed.sites[0].target:1.}
                    edits['record_key_changes_answer'] +=1
                    reordered=dict(w)
                    reordered['order']=tuple(reversed(w['order']))
                    same=realize_record(reordered)
                    assert same.sites[0].target == doc.sites[0].target
                    edits['record_order_preserves_answer'] +=1
                if kind == 'score_report':
                    w=doc.latent
                    assert score_holdout(w['initial'][0],w['increment']) == (
                        split=='composition')
                    side=('left' if w['query_team']==w['teams'][0] else 'right')
                    shortcuts['score_query_'+side] +=1
                    shortcuts['score_leaked' if w['headline_leak']
                              else 'score_unleaked'] +=1
                    changed=dict(w)
                    changed['increment']=1 if w['increment']!=1 else 2
                    edit_doc=realize_score(changed)
                    if side=='left':
                        assert edit_doc.sites[0].target != doc.sites[0].target
                        edits['left_score_event_changes_answer'] +=1
                    else:
                        assert edit_doc.sites[0].target == doc.sites[0].target
                        edits['right_score_event_preserves_answer'] +=1
                    assert site_oracle(edit_doc,edit_doc.sites[0]) == {
                        edit_doc.sites[0].target:1.}
                    toggled=dict(w)
                    toggled['headline_leak']=not w['headline_leak']
                    assert realize_score(toggled).sites[0].target == doc.sites[0].target
                    edits['headline_toggle_preserves_answer'] +=1
        result[split]=dict(n_per_kind=n_per_kind,total_sites=sum(families.values()),
                           sites=dict(families),shortcuts=dict(shortcuts),
                           paired_edits=dict(edits))
    overlap={kind:len(exact[('train',kind)] & exact[('composition',kind)])
             for kind in ('record_listing','score_report')}
    assert all(v==0 for v in overlap.values())
    lexical_rng=random.Random(seed+2)
    lexical=collections.Counter()
    lexical_kinds=collections.Counter()
    while sum(lexical_kinds.values())<49:
        doc=sample_document(lexical_rng,split='train')
        words=[token.lower() for token in doc.tokens if token.isalpha()]
        if len(words)>=100:
            lexical.update(words[:100])
            lexical_kinds[doc.kind]+=1
    return dict(version='document_mixture_v6', seed=seed, splits=result,
                train_composition_exact_overlap=overlap,
                first49_x100_alpha=dict(kinds=dict(lexical_kinds),
                                      types=len(lexical),
                                      singletons=sum(v==1 for v in lexical.values()),
                                      top10_share=sum(v for _,v in lexical.most_common(10))/4900),
                split_rule='score (initial_left + 2*increment) mod 5; '
                           'record ship/category assignment hash-like fold',
                caveats=['different narrative/event draws are not task-family holdouts',
                         'small lexicon and fixed score/record grammar persist',
                         'only V14 arm has exact full next-token q'])


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--n-per-kind',type=int,default=200)
    p.add_argument('--seed',type=int,default=23012)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    result=audit(a.n_per_kind,a.seed)
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    print('sites',{k:v['total_sites'] for k,v in result['splits'].items()},
          'overlap',result['train_composition_exact_overlap'],flush=True)

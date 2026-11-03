"""Audit visible-state recovery and causal edits of variable-clause documents."""

import argparse
import copy
import json
import random
import re
from collections import Counter
from pathlib import Path

from document_world import NAMES, VALUES
from document_world_v5_clause_realizations import (
    answer_from_visible_prefix, draw_varied, realize,
)


WORD=re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def _edited_world(world, plan):
    after,family,obj,attr=plan
    changed=copy.deepcopy(world)
    if family=='alias_identity':
        other=next(item for item in changed['initial'] if item!=obj)
        changed['aliases'][obj],changed['aliases'][other]=(
            changed['aliases'][other],changed['aliases'][obj])
        index=changed['recap_plan'].index(plan)
        changed['recap_plan'][index]=(after,family,other,attr)
    elif family=='source_attribution':
        j=next(j for j in range(after,-1,-1)
               if changed['events'][j]['object']==obj and
               changed['events'][j]['attribute']==attr)
        old=changed['sources'][j]
        changed['sources'][j]=next(name for name in NAMES if name!=old)
    else:
        attr=family.removeprefix('world_')
        earlier=[j for j in range(after,-1,-1)
                 if changed['events'][j]['object']==obj and
                 changed['events'][j]['attribute']==attr]
        if earlier:
            j=earlier[0]  # Most recent relevant event before this recap.
            old=changed['events'][j]['value']
            changed['events'][j]['value']=next(
                value for value in VALUES[attr] if value!=old)
        else:
            old=changed['initial'][obj][attr]
            changed['initial'][obj][attr]=next(
                value for value in VALUES[attr] if value!=old)
    return changed


def _check_sites(doc):
    for site in doc.sites:
        assert doc.tokens[site.index]==site.answer
        assert site.answer in site.candidates
        assert answer_from_visible_prefix(doc.tokens[:site.index])==site.answer


def audit(n,seed,style_entropy,aside_rate,paragraph_rate,header_rate,
          source_delay_rate=0.):
    rng=random.Random(seed)
    families=Counter();genres=Counter();edit_counts=Counter()
    lexical=Counter();eligible=0;word_count=[]
    changed_style=changed_aside=0
    for _ in range(n):
        doc=draw_varied(rng,style_entropy=style_entropy,
                        aside_rate=aside_rate,paragraph_rate=paragraph_rate,
                        header_rate=header_rate,
                        source_delay_rate=source_delay_rate)
        _check_sites(doc)
        genres[doc.genre]+=1
        families.update(site.family for site in doc.sites)
        words=[w.lower() for w in WORD.findall(doc.text())]
        word_count.append(len(words))
        if len(words)>=100 and eligible<49:
            lexical.update(words[:100]);eligible+=1
        for k,plan in enumerate(doc.latent['recap_plan']):
            edited=_edited_world(doc.latent,plan)
            other=realize(edited,style_entropy,aside_rate,paragraph_rate,
                          header_rate)
            assert other.sites[k].family==doc.sites[k].family
            assert other.sites[k].answer!=doc.sites[k].answer
            _check_sites(other)
            edit_counts[plan[1]]+=1
        # Whole-document surface changes must leave every world answer fixed.
        uniform=realize(doc.latent,1.,aside_rate,paragraph_rate,header_rate)
        fixed=realize(doc.latent,0.,aside_rate,paragraph_rate,header_rate)
        _check_sites(uniform);_check_sites(fixed)
        assert [(s.family,s.answer) for s in uniform.sites]==[
            (s.family,s.answer) for s in fixed.sites]
        changed_style+=uniform.tokens!=fixed.tokens
        no_aside=realize(doc.latent,style_entropy,0.,paragraph_rate,header_rate)
        with_aside=realize(doc.latent,style_entropy,.8,paragraph_rate,
                          header_rate)
        _check_sites(no_aside);_check_sites(with_aside)
        assert [(s.family,s.answer) for s in no_aside.sites]==[
            (s.family,s.answer) for s in with_aside.sites]
        changed_aside+=no_aside.tokens!=with_aside.tokens
    assert changed_style and changed_aside
    return dict(n=n,seed=seed,style_entropy=style_entropy,
                aside_rate=aside_rate,paragraph_rate=paragraph_rate,
                header_rate=header_rate,
                genres=dict(genres),sites_by_family=dict(families),
                paired_upstream_edits=dict(edit_counts),
                documents_changed_by_style=changed_style,
                documents_changed_by_aside=changed_aside,
                word_count=dict(min=min(word_count),
                                median=sorted(word_count)[len(word_count)//2],
                                max=max(word_count)),
                first49_eligible=eligible,
                first49_x100_alpha_types=len(lexical),
                first49_x100_alpha_top10_share=(
                    sum(v for _,v in lexical.most_common(10))/sum(lexical.values())
                    if lexical else None),
                caveat='Visible-prefix semantic recovery and paired-edit audit; '
                       'no full next-token oracle, learned model, or natural '
                       'language realism claim.')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--n',type=int,default=300)
    p.add_argument('--seed',type=int,default=33001)
    p.add_argument('--style-entropy',type=float,default=.7)
    p.add_argument('--aside-rate',type=float,default=.3)
    p.add_argument('--paragraph-rate',type=float,default=.22)
    p.add_argument('--header-rate',type=float,default=.5)
    p.add_argument('--source-delay-rate',type=float,default=0.)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    rows=[audit(args.n,args.seed+i,args.style_entropy,
                args.aside_rate,args.paragraph_rate,args.header_rate,
                args.source_delay_rate)
          for i in (0,1)]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(
        config={**vars(args),'output':str(args.output)},panels=rows),
        indent=2)+'\n')
    for row in rows:
        print(row['seed'],'sites',row['sites_by_family'],
              'types',row['first49_x100_alpha_types'],
              'top10',row['first49_x100_alpha_top10_share'],flush=True)


if __name__=='__main__':main()

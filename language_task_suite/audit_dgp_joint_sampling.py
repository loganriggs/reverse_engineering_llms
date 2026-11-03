"""Measure which rule families actually coexist in sampled toy-language streams.

This audit distinguishes the original mixed-rule generator from the later v8
stream and the separate answer-only task suite. It makes no FineWeb realism
claim; its purpose is to expose task density and compositional overlap.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from surface_mixture_v2 import _split
from surface_mixture_v4 import _content
from surface_mixture_v8 import SurfaceMixtureV8


STARTS=('As','Who','Section','Detail','What','Where','How')
FAMILIES=('mixed_verb','mixed_object','question_answer','source_answer',
          'scope_answer','count_answer')


def v8_panel(n,length,seed,kwargs):
    gen=SurfaceMixtureV8(seed=seed,**kwargs)
    labels=Counter()
    clauses=Counter()
    stream_families=Counter()
    pairwise=Counter()
    punctuation=Counter()
    examples=[]
    for i in range(n):
        ids,labs,qs=gen.sample_sequence(length)
        words=[gen.id2token[x] for x in ids]
        assert len(ids)==len(labs)==len(qs)==length
        for token,q in zip(words,qs):
            assert abs(sum(q.values())-1)<1e-8 and q.get(token,0)>0
        labels.update(labs)
        present={family for family in FAMILIES if family in labs}
        stream_families.update(present)
        for a in sorted(present):
            for b in sorted(present):
                if a<b:
                    pairwise[f'{a}+{b}']+=1
        history,_=_split(words)
        for raw in history:
            content=_content(raw)
            if content:
                clauses[content[0] if content[0] in STARTS else 'narrative']+=1
        punctuation.update(token for token in words if token in
                           ('(',')','[',']','"','?',':','\n'))
        if i<3:
            examples.append(' '.join(words))
    return dict(config=dict(n=n,length=length,seed=seed,kwargs=kwargs),
                labels=dict(labels),streams_with_family=dict(stream_families),
                streams_with_pair=dict(pairwise),
                completed_clause_types=dict(clauses),
                punctuation=dict(punctuation),examples=examples)


def original_panel(n,length,seed,source):
    sys.path.insert(0,str(source))
    from generator import ALL_RULES,DEFAULT_MIXING,LanguageGenerator
    gen=LanguageGenerator(ALL_RULES,DEFAULT_MIXING,mode='mixed',seed=seed)
    labels=Counter()
    streams=Counter()
    examples=[]
    for i in range(n):
        ids,labs=gen.sample_sequence(length)
        labels.update(labs)
        families=set()
        for label in labs:
            if label.startswith('skip_'):
                families.add('skip_bigram')
            elif label.startswith('bigram_'):
                families.add('bigram')
            elif label.startswith('trigram_'):
                families.add('trigram')
            elif label.startswith('induction_'):
                families.add('induction')
            elif label.startswith('bracket_'):
                families.add('bracket')
        streams.update(families)
        if i<3:
            examples.append(' '.join(gen.id2token[x] for x in ids))
    return dict(config=dict(n=n,length=length,seed=seed),labels=dict(labels),
                streams_with_family=dict(streams),examples=examples)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--n',type=int,default=300)
    p.add_argument('--length',type=int,default=256)
    p.add_argument('--seed',type=int,default=93001)
    p.add_argument('--source',default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    baseline=dict(counterbalanced_queries=True,question_rate=.05,
                  count_rate=0.,quote_rate=.08,source_rate=.15,
                  section_rate=.4,detail_rate=.8,scope_question_rate=.5)
    low=dict(baseline,section_rate=.06,scope_question_rate=.08)
    heavy=dict(baseline,quote_rate=.30,source_rate=.35)
    output=dict(config=vars(a),original=original_panel(a.n,a.length,a.seed,
                 Path(a.source)),v8_training=v8_panel(a.n,a.length,a.seed,
                 baseline),v8_low_section=v8_panel(a.n,a.length,a.seed,low),
                 v8_source_heavy=v8_panel(a.n,a.length,a.seed,heavy),
                 caveat='The original and v8 generators have different vocabularies '
                        'and document grammars. Counts measure their own sampled '
                        'processes, not matched task difficulty or natural-text '
                        'prevalence. The separate 12-family answer-only suite is '
                        'not sampled from either mixed stream.')
    path=Path(a.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(output,indent=2)+'\n')


if __name__=='__main__':main()

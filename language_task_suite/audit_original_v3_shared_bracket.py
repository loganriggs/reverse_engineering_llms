"""Audit joint-rule coverage and opener-swap support of the v3 successor."""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

_pre=argparse.ArgumentParser(add_help=False)
_pre.add_argument('--source',default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
_source,_=_pre.parse_known_args()
sys.path.insert(0,_source.source)

from original_v3_shared_bracket import (LanguageV3SharedBracket,
                                        SHARED_BRACKET_CONTENT,
                                        bracket_close_distribution,
                                        bracket_next_distribution)


def content_oracle_checks(seed,draws=20000):
    rows={}
    for opener,kind in (('(','paren'),('[','quote')):
        for content in (['big','alice'],['big','alice','old']):
            gen=LanguageV3SharedBracket(seed=seed+len(content)+(0 if opener=='(' else 100))
            oracle=bracket_next_distribution(gen,opener,content)
            continuation=sum(p for token,p in oracle.items() if token not in (')',']'))
            expected={token:p/continuation for token,p in oracle.items()
                      if token not in (')',']')}
            observed=Counter(gen._sample_bracket_content(kind,
                gen.token2cat[content[-1]],content[-1])[0] for _ in range(draws))
            tv=.5*sum(abs(observed[token]/draws-expected.get(token,0.))
                       for token in set(observed)|set(expected))
            assert tv<.03,(opener,content,tv)
            rows[f'{opener}:{"-".join(content)}']=dict(draws=draws,tv=tv)
    return rows


def panel(gen, nstreams, length):
    labels=Counter()
    close_counts=Counter()
    risk=Counter()
    opportunities=defaultdict(lambda:Counter())
    examples=[]
    triple=0
    for stream in range(nstreams):
        ids,labs=gen.sample_sequence(length)
        words=[gen.id2token[i] for i in ids]
        labels.update(labs)
        triple+=all(rule in labs for rule in
                    ('noun_bigram','skip_bigram','skip_trigram'))
        for t,token in enumerate(words):
            # Discard right-censored spans: otherwise short spans are
            # overrepresented near the end of a fixed-length stream.
            if token not in ('(', '[') or t+5>=length:
                continue
            closer=')' if token=='(' else ']'
            closers=[j for j in range(t+1,min(length,t+6)) if words[j]==closer]
            if not closers:
                continue
            end=closers[0]
            content=words[t+1:end]
            if not 2<=len(content)<=4 or labs[end]!='skip_trigram':
                continue
            # The first content word and every subsequent category transition
            # have positive likelihood under *either* opener by construction.
            cats=[gen.token2cat[w] for w in content]
            assert cats[0]==SHARED_BRACKET_CONTENT['start_cat']
            for left,right in zip(cats,cats[1:]):
                assert any(name==right and probability>0 for name,probability in
                           SHARED_BRACKET_CONTENT['transitions'][left])
            for count in range(2,len(content)+1):
                next_token=words[t+count+1]
                opportunities[(token,count)]['total']+=1
                opportunities[(token,count)]['close']+=(next_token==closer)
            close_counts[(token,len(content))]+=1
            old=bracket_close_distribution(token,len(content))
            new=bracket_close_distribution('[' if token=='(' else '(',len(content))
            assert abs(sum(old.values())-sum(new.values()))<1e-12
            assert next(iter(old))!=next(iter(new))
            risk['supported_opener_swap_events']+=1
            if len(examples)<6:
                examples.append(dict(original=' '.join(words[t:end]),
                                     edited=' '.join((['[' if token=='(' else '(']
                                                      + content)),
                                     old_closer=closer,new_closer=next(iter(new)),
                                     closer_mass=next(iter(old.values()))))
    frequency={f'{opener}:{count}':dict(total=row['total'],close=row['close'],
                   empirical=row['close']/row['total'],
                   predicted=next(iter(bracket_close_distribution(opener,count).values())))
               for (opener,count),row in opportunities.items()}
    return dict(config=dict(nstreams=nstreams,length=length),labels=dict(labels),
                streams_with_noun_skip_bracket=triple,
                close_counts={f'{o}:{n}':v for (o,n),v in close_counts.items()},
                supported_opener_swap_events=risk['supported_opener_swap_events'],
                conditional_close_frequency=frequency,examples=examples)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,
                   default=Path('/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits'))
    p.add_argument('--nstreams',type=int,default=4000)
    p.add_argument('--length',type=int,default=129)
    p.add_argument('--seed',type=int,default=97001)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    gen=LanguageV3SharedBracket(seed=a.seed)
    result=dict(config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
                panel=panel(gen,a.nstreams,a.length),
                content_oracle_checks=content_oracle_checks(a.seed),
                caveat='Shared content makes opener-only edits of the visible '
                       'pre-closer prefix supported. It does not make the '
                       'stylized grammar naturalistic or identify the full '
                       'visible-prefix distribution outside bracket spans.')
    for key,row in result['panel']['conditional_close_frequency'].items():
        if row['total']>=100:
            assert abs(row['empirical']-row['predicted'])<.03,(key,row)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(result['panel']['supported_opener_swap_events'],
          result['panel']['streams_with_noun_skip_bracket'])
    print(result['panel']['conditional_close_frequency'])


if __name__=='__main__':main()

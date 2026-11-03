"""Audit local conditional action and inherited cross-sentence V13 rules."""

import argparse
import json
from collections import Counter
from pathlib import Path

from audit_v13_attributes import panel as inherited_panel
from surface_mixture_v2 import _split
from surface_mixture_v4 import _content
from surface_mixture_v13_attributes import ATTRIBUTES, attribute_state
from surface_mixture_v14_conditional import (
    ACTION, SurfaceMixtureV14Conditional,
)
from train_v13_attributes import COMMON, DIAGNOSTIC, REFERENCE


def repair_completed_conditional(words,changed,attribute_pos,replacement,t):
    # Recover absolute clause boundaries from completed periods, including
    # optional newline/number prefixes.
    clause_start=0
    for i,word in enumerate(words[:attribute_pos]):
        if word=='.':
            clause_start=i+1
    clause_end=next((i for i in range(attribute_pos,len(words))
                     if words[i]=='.'),None)
    if clause_end is None or clause_end>=t:
        return False
    raw=words[clause_start:clause_end+1]
    content=_content(raw)
    if not (content and content[0]=='The' and content[2:3]==['is']
            and len(content)==10):
        return False
    the_pos=clause_start+raw.index('The')
    assert attribute_pos==the_pos+3
    action_pos=the_pos+7
    assert action_pos<t and words[action_pos]==ACTION[content[3]]
    changed[action_pos]=ACTION[replacement]
    return True


def conditional_panel(seed,n,length,kwargs,
                      generator_cls=SurfaceMixtureV14Conditional):
    gen=generator_cls(seed=seed,**kwargs)
    metrics,labels=Counter(),Counter()
    examples=[]
    for stream in range(n):
        ids,labs,qs=gen.sample_sequence(length)
        words=[gen.id2token[i] for i in ids]
        labels.update(labs)
        kinds=set(labs)
        metrics['streams_conditional_action'] += 'conditional_action' in kinds
        metrics['streams_conditional_and_attribute_object'] += all(
            v in kinds for v in ('conditional_action','attribute_object'))
        metrics['streams_conditional_and_bracket'] += all(
            v in kinds for v in ('conditional_action','bracket_close'))
        metrics['streams_conditional_and_price_answer'] += all(
            v in kinds for v in ('conditional_action','price_answer'))
        for t,(word,q,label) in enumerate(zip(words,qs,labs)):
            assert word in q and q[word]>0
            assert abs(sum(q.values())-1)<1e-9
            assert q==gen.next_distribution(words[:t])
            if label!='conditional_action':
                continue
            history,raw=_split(words[:t]);content=_content(raw)
            assert content[:1]==['The'] and len(content)==7
            assert content[2]=='is' and content[4:6]==[',','so']
            attribute=content[3]
            assert q=={ACTION[attribute]:1.}
            metrics['conditional_action_sites'] += 1
            metrics['conditional_attribute_'+attribute] += 1
            prior=attribute_state(history)['facts']
            replacement=next((v for v in ATTRIBUTES
                              if v not in prior.values() and v!=attribute),None)
            if replacement is None:
                continue
            item=content[1]
            position=t-len(raw)+raw.index('The')+3
            assert words[position]==attribute
            changed=words[:t].copy();changed[position]=replacement
            assert gen.next_distribution(words[:position]).get(replacement,0)>0
            for u in range(position+1,t):
                assert gen.next_distribution(changed[:u]).get(changed[u],0)>0
            assert gen.next_distribution(changed)=={ACTION[replacement]:1.}
            metrics['supported_conditional_edits'] += 1
            if len(examples)<4:
                examples.append(dict(item=item,old_attribute=attribute,
                                     new_attribute=replacement,
                                     old_action=word,new_action=ACTION[replacement]))
    return dict(seed=seed,n=n,length=length,metrics=dict(metrics),
                labels=dict(labels),examples=examples)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,default=99641)
    p.add_argument('--n',type=int,default=100)
    p.add_argument('--length',type=int,default=256)
    p.add_argument('--regime',choices=('reference','diagnostic'),default='reference')
    p.add_argument('--conditional-rate',type=float,default=.35)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if not 0 <= args.conditional_rate < 1:
        p.error('--conditional-rate must be in [0, 1)')
    kwargs=dict(COMMON,**(REFERENCE if args.regime=='reference' else DIAGNOSTIC),
                conditional_continuation_rate=args.conditional_rate)
    rows=[]
    for seed in (args.seed,args.seed+1):
        inherited=inherited_panel(seed,args.n,args.length,kwargs,
                                  SurfaceMixtureV14Conditional,
                                  repair_completed_conditional)
        conditional=conditional_panel(seed,args.n,args.length,kwargs)
        assert inherited['labels']==conditional['labels']
        rows.append(dict(seed=seed,inherited=inherited,
                         conditional=conditional))
        print(seed,'inherited',inherited['metrics'],
              'conditional',conditional['metrics'],flush=True)
    result=dict(config={**vars(args),'output':str(args.output),
                        'generator_kwargs':kwargs},panels=rows,
                caveat='Exact sampled-prefix and supported-edit audit of a '
                       'synthetic conditional continuation. It does not '
                       'establish natural-language realism or model competence.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()

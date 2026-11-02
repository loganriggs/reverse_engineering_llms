"""Independently audit supported two-section curriculum prefixes."""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from pretrain_v8_scope_lookup import generator, make_world


def independent_facts(tokens):
    """Parse heading/detail word patterns without the generator's state code."""
    facts={}
    active=None
    used=set()
    for i,word in enumerate(tokens):
        if (word=='Section' and tokens[i+2]==':' and tokens[i+4]=='.'):
            active=tokens[i+1]
            assert active not in used
            used.add(active)
        elif (word=='Detail' and tokens[i+1:i+5]==[':','the','badge','was']):
            assert active is not None
            facts[active]=tokens[i+5]
            active=None
    assert active is None
    return facts,used


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--worlds',type=int,default=1000)
    p.add_argument('--seed',type=int,default=47001)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    gen=generator(a.seed)
    rng=random.Random(a.seed+1)
    counts=Counter()
    lengths=[]
    examples=[]
    for _ in range(a.worlds):
        pair=make_world(gen,rng)
        assert len(pair)==2
        keys=[]
        for tokens,answer in pair:
            facts,used=independent_facts(tokens)
            content=tokens[-11:]
            key=content[7]
            assert content[:7]==['What','color','was','the','badge','in','section']
            assert content[8:]==['?','Answer',':']
            assert len(facts)==2 and len(used)==2
            assert facts[key]==answer
            assert gen.next_distribution(tokens)=={answer:1.}
            keys.append(key)
            lengths.append(len(tokens))
            counts[f'key_{key}']+=1
            counts[f'answer_{answer}']+=1
            for i,word in enumerate(tokens):
                assert gen.next_distribution(tokens[:i]).get(word,0)>0
        assert keys[0]!=keys[1] and pair[0][1]!=pair[1][1]
        counts['query_switches_answer']+=1
        if len(examples)<3:
            examples.append(dict(prefix=' '.join(pair[0][0]).replace(' \n ','\n'),
                                 answer=pair[0][1],alternative_answer=pair[1][1]))
    output=dict(config=vars(a),counts=dict(counts),
                min_prefix_length=min(lengths),max_prefix_length=max(lengths),
                mean_prefix_length=sum(lengths)/len(lengths),examples=examples,
                caveat='Audit checks exact prefix support and independently '
                       'reconstructs visible section facts. It does not '
                       'validate natural-language prevalence or mixed-stream competence.')
    path=Path(a.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(output,indent=2)+'\n')
    print('audited',a.worlds,'worlds','length',output['min_prefix_length'],
          output['max_prefix_length'],flush=True)


if __name__=='__main__':
    main()

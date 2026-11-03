"""Classify why `the` dominates V14's matched word windows."""

import argparse
import json
from collections import Counter
from pathlib import Path

from audit_surface_repetition import WORD
from surface_mixture_v2 import _frame, _split
from surface_mixture_v4 import _content
from surface_mixture_v12_price_faq import SPECIAL
from surface_mixture_v14_conditional import SurfaceMixtureV14Conditional
from train_v13_attributes import COMMON, REFERENCE


PLACE_EVENT_DETERMINERS={
    'lead':{8,11},'subject':{3,11},'at':{1,10},
    'when':{1,11},'in_place':{1,9},'during':{1,9},
}
OBJECT_DETERMINERS={
    'lead':4,'subject':7,'at':6,'when':7,'in_place':5,'during':5,
}


def classify(words,t):
    _,raw=_split(words[:t])
    content=_content(raw)
    if not content or content[0] in SPECIAL:
        return 'special_or_sentence_start'
    frame=_frame(content)
    pos=len(content)
    if pos in PLACE_EVENT_DETERMINERS[frame]:
        return 'narrative_place_or_event_determiner'
    if pos==OBJECT_DETERMINERS[frame]:
        return 'narrative_object_determiner'
    return 'narrative_tail_or_other'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--n',type=int,default=480)
    p.add_argument('--words',type=int,default=100)
    p.add_argument('--length',type=int,default=256)
    p.add_argument('--seed',type=int,default=99851)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    gen=SurfaceMixtureV14Conditional(seed=a.seed,**COMMON,**REFERENCE,
                                     conditional_continuation_rate=.35)
    the_slots=Counter();all_words=Counter()
    for _ in range(a.n):
        ids,_,_=gen.sample_sequence(a.length)
        tokens=[gen.id2token[i] for i in ids]
        seen=0
        for t,token in enumerate(tokens):
            if not WORD.fullmatch(token):
                continue
            seen+=1
            if seen>a.words:break
            all_words[token.lower()]+=1
            if token.lower()=='the':
                the_slots[classify(tokens,t)]+=1
        assert seen>a.words
    total=a.n*a.words
    assert sum(all_words.values())==total
    assert sum(the_slots.values())==all_words['the']
    result=dict(config={**vars(a),'output':str(a.output)},
                total_words=total,the_count=all_words['the'],
                the_mass=all_words['the']/total,
                a_count=all_words['a'],an_count=all_words['an'],
                slots={key:dict(count=value,mass=value/total)
                       for key,value in sorted(the_slots.items())},
                minimum_the_mass_if_only_place_event_slots_changed=(
                    all_words['the']-
                    the_slots['narrative_place_or_event_determiner'])/total,
                caveat='Position classes are based on V14 fixed-index '
                       'frames. This is an intervention feasibility count, '
                       'not a valid grammatical replacement proposal; '
                       'changing articles also changes next-word support '
                       'and task frequencies.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()

"""Oracle, variable-distance, and counterfactual audit for nested spans."""

import argparse
import json
from collections import Counter
from pathlib import Path

from surface_mixture_v2 import EVENTS, PLACES, _frame, _indices, _split
from surface_mixture_v2_nested import CLOSERS, OPENERS, render_nested
from surface_mixture_v2_nested_varlen import SurfaceMixtureV2NestedVarlen
from surface_mixture_v2_varied import SurfaceMixtureV2Varied


def stack_at(prefix):
    stack = []
    for pos, token in enumerate(prefix):
        if token in OPENERS:
            stack.append((pos, token))
        elif token in CLOSERS.values():
            assert stack and CLOSERS[stack[-1][1]] == token
            stack.pop()
    return stack


def supported(gen, prefix):
    for i, token in enumerate(prefix):
        if gen.next_distribution(prefix[:i]).get(token, 0) <= 0:
            return False
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=1000)
    p.add_argument('--length', type=int, default=128)
    p.add_argument('--seed', type=int, default=20261202)
    p.add_argument('--nested-mass', type=float, default=.3)
    p.add_argument('--inner-mass', type=float, default=.6)
    p.add_argument('--extra-mass', type=float, default=.5)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    gen = SurfaceMixtureV2NestedVarlen(seed=a.seed,
        nested_mass=a.nested_mass,inner_mass=a.inner_mass,
        extra_mass=a.extra_mass)
    base = SurfaceMixtureV2Varied(seed=a.seed,parenthetical_mass=0.)
    labels, outer_lags, inner_lags, pairs = (Counter() for _ in range(4))
    verb_sites = support_checks = 0
    examples = []
    for stream in range(a.n):
        ids,labs,qs=gen.sample_sequence(a.length)
        tokens=[gen.id2token[i] for i in ids]
        if stream < 5:
            examples.append(render_nested(ids))
        for t,(token,label,q) in enumerate(zip(tokens,labs,qs)):
            prefix=tokens[:t]
            assert q==gen.next_distribution(prefix)
            assert token in q and q[token]>0 and abs(sum(q.values())-1)<1e-12
            labels[label]+=1
            if label=='mixed_verb':
                assert q==base.next_distribution(prefix)
                verb_sites+=1
            if label not in ('nested_close_inner','nested_close_outer'):
                continue
            _,clause=_split(prefix)
            suffix=clause[_indices(_frame(clause))[3]:]
            stack=stack_at(suffix)
            assert len(q)==1 and len(stack)==(2 if label.endswith('inner') else 1)
            assert q=={CLOSERS[stack[-1][1]]:1.}
            lag=len(suffix)-stack[-1][0]
            (inner_lags if label.endswith('inner') else outer_lags)[lag]+=1
            if label.endswith('outer'):
                assert len(suffix)>=2 and len(stack_at(suffix))==1
                openers=[x for x in suffix if x in OPENERS]
                if len(openers)==2:
                    pairs[''.join(openers)]+=1
            relevant=suffix.copy()
            rel_pos=stack[-1][0]
            relevant[rel_pos]=next(x for x in OPENERS if x!=relevant[rel_pos])
            changed=prefix[:-len(suffix)]+relevant
            assert gen.next_distribution(changed)=={
                CLOSERS[relevant[rel_pos]]:1.}
            if label.endswith('inner'):
                control=suffix.copy()
                control[stack[0][0]]=next(x for x in OPENERS
                                               if x!=control[stack[0][0]])
            else:
                control=suffix.copy()
                options=[]
                for i in range(1,len(suffix)-2):
                    if suffix[i] in ('near','during') and suffix[i+1]=='the':
                        vocab=PLACES if suffix[i]=='near' else EVENTS
                        if suffix[i+2] in vocab:
                            options.append((i+2,vocab))
                assert options
                pos,vocab=options[-1]
                control[pos]=next(x for x in vocab if x!=control[pos])
            unchanged=prefix[:-len(suffix)]+control
            assert gen.next_distribution(unchanged)==q
            if support_checks<400:
                assert supported(gen,changed) and supported(gen,unchanged)
                support_checks+=2
        history,_=_split(tokens)
        for clause in history:
            suffix=clause[_indices(_frame(clause))[3]:]
            if suffix and suffix[0] in OPENERS:
                assert suffix[-1]=='.' and not stack_at(suffix[:-1])
    if a.n >= 100:
        assert len(outer_lags)>=3 and inner_lags[8]>0
    result=dict(config={**vars(a),'output':str(a.output)},
                vocab_size=len(gen.vocab),verb_sites=verb_sites,
                labels=dict(labels),outer_closer_lags=dict(outer_lags),
                inner_closer_lags=dict(inner_lags),
                nested_opener_pairs=dict(pairs),
                edited_prefix_support_checks=support_checks,
                examples=examples,
                caveat='Visible-prefix oracle, opener counterfactuals, and '
                       'all complete bracket stacks pass independent checks. '
                       'Variable lag rules out a single fixed-offset copy, '
                       'but does not prove a model uses an abstract stack.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('verb_sites','labels',
        'outer_closer_lags','inner_closer_lags','nested_opener_pairs')},indent=2))


if __name__=='__main__':
    main()

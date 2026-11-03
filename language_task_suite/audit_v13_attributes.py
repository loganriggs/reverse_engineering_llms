"""Audit the V13 shared attribute fact → narrative description dependency."""

import argparse
import json
from collections import Counter
from pathlib import Path

from surface_mixture_v2 import _frame, _split
from surface_mixture_v3 import OBJECT_INDEX
from surface_mixture_v4 import _content
from surface_mixture_v12_price_faq import PRODUCTS, price_state
from surface_mixture_v13_attributes import (
    ATTRIBUTES, SurfaceMixtureV13Attributes, attribute_state,
)


def attribute_positions(words):
    out = {}
    start = 0
    for i, token in enumerate(words):
        if token != '.':
            continue
        raw = words[start:i+1]
        content = _content(raw)
        if content and content[0] == 'The' and content[2:3] == ['is']:
            out[content[1]] = (start+raw.index('The')+3, content[3])
        start = i+1
    return out


def panel(seed, n, length, kwargs):
    gen = SurfaceMixtureV13Attributes(seed=seed, **kwargs)
    metrics, labels = Counter(), Counter()
    pairs = Counter()
    examples = []
    for stream in range(n):
        ids, labs, qs = gen.sample_sequence(length)
        words = [gen.id2token[i] for i in ids]
        labels.update(labs)
        kinds = set(labs)
        for kind in ('attribute_fact_value', 'attribute_adjective',
                     'attribute_object', 'price_answer', 'source_answer',
                     'mixed_verb', 'bracket_close'):
            metrics['streams_'+kind] += kind in kinds
        metrics['streams_attribute_fact_and_descriptive_object'] += (
            'attribute_fact_value' in kinds and 'attribute_object' in kinds)
        metrics['streams_attribute_object_and_bracket'] += (
            'attribute_object' in kinds and 'bracket_close' in kinds)
        positions = attribute_positions(words)
        for t, (word, q, label) in enumerate(zip(words, qs, labs)):
            assert abs(sum(q.values())-1) < 1e-9
            assert word in q and q[word] > 0
            assert q == gen.next_distribution(words[:t])
            history, raw = _split(words[:t])
            content = _content(raw)
            if not raw and history:
                n_price=len(price_state(history)['facts'])
                n_attribute=len(attribute_state(history)['facts'])
                p=(gen.price_fact_rate if n_price<len(PRODUCTS) else 0.)
                a=(gen.attribute_fact_rate
                   if n_attribute<len(PRODUCTS) else 0.)
                expected=(1-gen.newline_rate)*(p*(1-a)+a)
                assert abs(q.get('The',0.)-expected)<1e-10
                metrics['checked_fact_start_mixtures'] += 1
            if label == 'fact_kind':
                item=content[1]
                prices=price_state(history)['facts']
                attributes=attribute_state(history)['facts']
                p=(gen.price_fact_rate if len(prices)<len(PRODUCTS) else 0.)
                a=(gen.attribute_fact_rate if len(attributes)<len(PRODUCTS)
                   else 0.)
                pp=(1-a)*p/(len(PRODUCTS)-len(prices)) if item not in prices else 0.
                aa=a/(len(PRODUCTS)-len(attributes)) if item not in attributes else 0.
                assert abs(q.get('costs',0.)-pp/(pp+aa))<1e-10
                assert abs(q.get('is',0.)-aa/(pp+aa))<1e-10
                metrics['checked_fact_kind_posteriors'] += 1
            if label not in ('attribute_adjective', 'attribute_object'):
                continue
            state = attribute_state(history)
            pending = [(item, adjective)
                       for item, adjective in state['facts'].items()
                       if item not in state['mentioned']]
            assert pending
            if label == 'attribute_adjective':
                item, adjective = pending[0]
            else:
                matches = [(item, adjective) for item, adjective in pending
                           if content[-1] == adjective]
                assert len(matches) == 1
                item, adjective = matches[0]
            pos, old = positions[item]
            assert old == adjective and pos < t
            if label == 'attribute_adjective':
                assert len(content) == OBJECT_INDEX[_frame(content)]-1
                metrics['attribute_adjective_sites'] += 1
                metrics['attribute_adjective_multi_pending_sites'] += len(pending)>1
                pairs[(item, adjective)] += 1
            else:
                assert len(content) == OBJECT_INDEX[_frame(content)]
                assert content[-1] == adjective
                metrics['attribute_object_sites'] += 1
                metrics['attribute_object_multi_pending_sites'] += len(pending)>1
                parts = gen.object_rule_components(words[:t])
                rebuilt = Counter()
                for part in parts.values():
                    for output, probability in part['distribution'].items():
                        rebuilt[output] += part['mass']*probability
                assert set(q) == set(rebuilt)
                assert max(abs(q[k]-v) for k,v in rebuilt.items()) < 1e-10
                assert abs(parts['attribute_fact']['mass']-
                           gen.attribute_object_mass) < 1e-10
            # An unused attribute changes the earlier fact while preserving
            # every intervening observed token's support under the oracle.
            replacement = next((v for v in ATTRIBUTES
                                if v not in state['facts'].values()), None)
            if replacement is None:
                continue
            changed = words[:t].copy()
            changed[pos] = replacement
            for u in range(pos+1,t):
                assert gen.next_distribution(changed[:u]).get(changed[u],0)>0, (
                    seed,stream,t,pos,u,changed[u])
            edited = gen.next_distribution(changed)
            if label == 'attribute_adjective':
                assert abs(edited[replacement]-q[replacement]-
                           gen.attribute_adjective_mass/len(pending)) < 1e-9
                metrics['supported_adjective_edits'] += 1
                if len(examples) < 4:
                    examples.append(dict(item=item,old_attribute=old,
                                         new_attribute=replacement,
                                         target_prefix=' '.join(content)))
            else:
                assert edited.get(item,0.) < q[item]
                metrics['supported_object_edits'] += 1
    return dict(seed=seed,n=n,length=length,kwargs=kwargs,
                metrics=dict(metrics),labels=dict(labels),
                item_attribute_pairs={'|'.join(k):v for k,v in sorted(pairs.items())},
                examples=examples)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n',type=int,default=100)
    p.add_argument('--length',type=int,default=256)
    p.add_argument('--seed',type=int,default=99621)
    p.add_argument('--regime',choices=('reference_candidate','diagnostic'),
                   default='reference_candidate')
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    kwargs = dict(counterbalanced_queries=True,question_rate=.05,
                  count_rate=0.,quote_rate=.08,source_rate=.15,
                  section_rate=.06,detail_rate=.8,scope_question_rate=.08,
                  source_object_mass=.15,source_subject_mass=.15,
                  bracket_rate=.06,linked_subject_mass=.65,
                  linked_object_mass=.65,linked_question_mass=.30,
                  linked_query_key_mass=.65)
    if args.regime == 'diagnostic':
        kwargs.update(price_fact_rate=.12,price_question_rate=.24,
                      price_object_mass=.65,attribute_fact_rate=.20,
                      attribute_adjective_mass=.80,attribute_object_mass=.80)
    else:
        kwargs.update(price_fact_rate=.06,price_question_rate=.12,
                      price_object_mass=.45,attribute_fact_rate=.10,
                      attribute_adjective_mass=.65,attribute_object_mass=.65)
    rows = [panel(args.seed+i,args.n,args.length,kwargs) for i in (0,1)]
    result = dict(config={**vars(args),'output':str(args.output)},panels=rows,
                  caveat='Rates are provisional. This checks the exact '
                         'visible-prefix oracle, supported edits and task '
                         'co-occurrence; it does not establish natural-language '
                         'realism or learned-model competence.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    for row in rows:
        print(row['seed'],row['metrics'],flush=True)


if __name__ == '__main__':
    main()

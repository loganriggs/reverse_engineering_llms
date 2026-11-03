"""Audit V12 exact support, linked documents, and price counterfactuals."""

import argparse
import json
from collections import Counter
from pathlib import Path

from surface_mixture_v2 import _frame, _indices, _split
from surface_mixture_v3 import OBJECT_INDEX
from surface_mixture_v4 import _content
from surface_mixture_v10_linked_events import linked_state
from surface_mixture_v12_price_faq import (PRICES, SurfaceMixtureV12PriceFAQ,
                                            price_state)


def fact_positions(words):
    positions = {}
    start = 0
    for i, token in enumerate(words):
        if token != '.':
            continue
        clause = words[start:i+1]
        content = _content(clause)
        if content and content[0] == 'The':
            positions[content[1]] = (start+clause.index('The')+3, content[3])
        start = i+1
    return positions


def panel(seed, n, length, kwargs):
    gen = SurfaceMixtureV12PriceFAQ(seed=seed, **kwargs)
    metrics, labels = Counter(), Counter()
    examples = []
    price_pairs = Counter()
    for stream in range(n):
        ids, labs, distributions = gen.sample_sequence(length)
        words = [gen.id2token[t] for t in ids]
        labels.update(labs)
        kinds = set(labs)
        for kind in ('price_fact_value','priced_item_object','price_query_key',
                     'price_answer','mixed_verb','mixed_object','bracket_close',
                     'source_answer','scope_answer'):
            metrics['streams_'+kind] += kind in kinds
        metrics['streams_all_price_fact_object_answer'] += all(
            k in kinds for k in ('price_fact_value','priced_item_object',
                                 'price_answer'))
        metrics['streams_price_answer_and_bracket'] += (
            'price_answer' in kinds and 'bracket_close' in kinds)
        positions = fact_positions(words)
        for t, (token, q, label) in enumerate(zip(words, distributions, labs)):
            assert abs(sum(q.values())-1) < 1e-9
            assert token in q and q[token] > 0
            assert q == gen.next_distribution(words[:t])
            if label == 'priced_item_object':
                parts = gen.object_rule_components(words[:t])
                rebuilt = Counter()
                for part in parts.values():
                    for output, probability in part['distribution'].items():
                        rebuilt[output] += part['mass']*probability
                assert set(q) == set(rebuilt)
                assert max(abs(q[k]-v) for k,v in rebuilt.items()) < 1e-10
                assert abs(parts['priced_item']['mass']-
                           gen.price_object_mass) < 1e-10
                metrics['audited_priced_object_components'] += 1
            if label != 'price_answer':
                continue
            history, raw = _split(words[:t])
            content = _content(raw)
            assert content[:4] == ['How','much','does','the']
            product = content[4]
            state = price_state(history)
            assert len(state['facts']) >= 2
            assert q == {state['facts'][product]:1.}
            assert product in positions
            metrics['audited_price_answers'] += 1
            price_pairs[(product, token)] += 1
            query_pos=t-len(raw)+raw.index('How')+4
            assert words[query_pos]==product
            alternatives=[p for p in state['facts'] if p!=product and
                          p not in state['answered']]
            if alternatives:
                other=alternatives[0]
                changed_key=words[:t].copy()
                changed_key[query_pos]=other
                assert gen.next_distribution(words[:query_pos]).get(other,0)>0
                for u in range(query_pos+1,t):
                    assert gen.next_distribution(changed_key[:u]).get(
                        changed_key[u],0)>0
                assert gen.next_distribution(changed_key)=={
                    state['facts'][other]:1.}
                metrics['support_preserving_query_key_edits'] += 1
            # Edit the earlier item's price to an unused value. Every
            # subsequent token in the edited prefix must retain support;
            # the answer must change at this same query position.
            replacement = next((v for v in PRICES if v not in
                                state['facts'].values()), None)
            if replacement is None:
                continue
            pos, old = positions[product]
            assert pos < t and old == token
            changed = words[:t].copy()
            changed[pos] = replacement
            for u in range(pos+1,t):
                assert gen.next_distribution(changed[:u]).get(changed[u],0)>0, (
                    stream,t,pos,u,changed[u])
            assert gen.next_distribution(changed) == {replacement:1.}
            metrics['support_preserving_price_edits'] += 1
            if len(examples)<4:
                examples.append(dict(stream=stream, product=product,
                    old_price=old, edited_price=replacement,
                    question=' '.join(content),
                    preceding_completed_clauses=len(history)))
        # Validate that this extension did not suppress the V10 linked
        # source state when a price fact or query appears in history.
        history,_ = _split(words)
        linked_state(history)
    return dict(seed=seed,n=n,length=length,kwargs=kwargs,
                metrics=dict(metrics),labels=dict(labels),
                price_pairs={'|'.join(k):v for k,v in sorted(price_pairs.items())},
                examples=examples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--length', type=int, default=256)
    parser.add_argument('--seed', type=int, default=99601)
    parser.add_argument('--regime', choices=('reference_candidate','diagnostic'),
                        default='reference_candidate')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    kwargs = dict(counterbalanced_queries=True,question_rate=.05,
                  count_rate=0.,quote_rate=.08,source_rate=.15,
                  section_rate=.06,detail_rate=.8,scope_question_rate=.08,
                  source_object_mass=.15,source_subject_mass=.15,
                  bracket_rate=.06,linked_subject_mass=.65,
                  linked_object_mass=.65,linked_question_mass=.30,
                  linked_query_key_mass=.65)
    if args.regime == 'diagnostic':
        kwargs.update(price_fact_rate=.18,price_question_rate=.40,
                      price_object_mass=.80)
    else:
        kwargs.update(price_fact_rate=.08,price_question_rate=.18,
                      price_object_mass=.55)
    rows = [panel(args.seed+i,args.n,args.length,kwargs) for i in (0,1)]
    output = dict(config={**vars(args),'output':str(args.output)},panels=rows,
      caveat='All rates are provisional synthetic choices. This audits exact '
             'visible-prefix support, rule components, and edited-prefix '
             'support, not similarity to FineWeb or learned-model competence.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2)+'\n')
    for row in rows:
        print(row['seed'],row['metrics'],flush=True)


if __name__ == '__main__':
    main()

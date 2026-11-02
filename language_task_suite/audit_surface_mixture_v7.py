"""Replay the v7 prefix oracle and test section-key causal counterfactuals."""

import argparse
import json
from collections import Counter
from pathlib import Path

from surface_mixture_v2 import _split
from surface_mixture_v4 import _content
from surface_mixture_v7 import SurfaceMixtureV7, render
from surface_mixture_v8 import SurfaceMixtureV8


def independent_facts(history):
    active = None
    facts = {}
    for raw in history:
        words = _content(raw)
        if words and words[0] == 'Section':
            active = words[1]
        elif words and words[0] == 'Detail':
            assert active is not None and active not in facts
            facts[active] = words[5]
            active = None
    return facts


def supported(gen, words):
    return all(gen.next_distribution(words[:t]).get(word, 0) > 0
               for t, word in enumerate(words))


def audit(n, length, seed, generator='v7', **kwargs):
    generator_class = {'v7': SurfaceMixtureV7, 'v8': SurfaceMixtureV8}[generator]
    gen = generator_class(seed=seed, counterbalanced_queries=True, **kwargs)
    counts = Counter()
    key_edits = 0
    answer_sites = 0
    section_intervening_narrative = 0
    examples = []
    for row_index in range(n):
        ids, labels, qs = gen.sample_sequence(length)
        words = [gen.id2token[i] for i in ids]
        if row_index < 4:
            examples.append(render(ids))
        for t, (word, label, q) in enumerate(zip(words, labels, qs)):
            assert q == gen.next_distribution(words[:t])
            assert abs(sum(q.values())-1.) < 1e-12
            assert q.get(word, 0) > 0
            counts[label] += 1
            if generator == 'v8' and word == 'Section':
                assert t > 0 and words[t-1] == '\n'
            if label != 'scope_answer':
                continue
            answer_sites += 1
            history, raw = _split(words[:t])
            content = _content(raw)
            facts = independent_facts(history)
            assert content[:7] == ['What', 'color', 'was', 'the',
                                    'badge', 'in', 'section']
            assert content[8:] == ['?', 'Answer', ':']
            assert q == {facts[content[7]]: 1.}
            assert gen.scope_answer(words[:t]) == facts[content[7]]
            key_pos = t-4
            key_dist = gen.next_distribution(words[:key_pos])
            assert key_dist == {key: 1/len(facts) for key in facts}
            for key, color in facts.items():
                changed = words[:t].copy()
                changed[key_pos] = key
                assert supported(gen, changed)
                assert gen.next_distribution(changed) == {color: 1.}
                if key != words[key_pos]:
                    assert color != word
                    key_edits += 1
        # Heading, then a narrative sentence, then a scoped detail.
        history, _ = _split(words)
        pending = False
        intervened = False
        for raw in history:
            content = _content(raw)
            if content and content[0] == 'Section':
                pending, intervened = True, False
            elif content and content[0] == 'Detail' and pending:
                section_intervening_narrative += int(intervened)
                pending = False
            elif content and pending and content[0] not in ('Where', 'How', 'Who', 'What', 'As'):
                intervened = True
    assert answer_sites and key_edits and section_intervening_narrative
    return dict(config=dict(n=n, length=length, seed=seed,
                            generator=generator, **kwargs),
                vocab_size=len(gen.vocab), labels=dict(counts),
                scope_answer_sites=answer_sites,
                full_prefix_supported_alternative_keys=key_edits,
                details_after_intervening_narrative=section_intervening_narrative,
                examples=examples,
                caveat='Exact synthetic oracle and support checks; no model '
                       'competence or natural-text realism is implied.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=1000)
    p.add_argument('--length', type=int, default=256)
    p.add_argument('--seed', type=int, default=12001)
    p.add_argument('--generator', choices=('v7', 'v8'), default='v7')
    p.add_argument('--section-rate', type=float, default=.18)
    p.add_argument('--detail-rate', type=float, default=.80)
    p.add_argument('--scope-question-rate', type=float, default=.40)
    p.add_argument('--quote-rate', type=float, default=.08)
    p.add_argument('--source-rate', type=float, default=.15)
    p.add_argument('--output', type=Path,
                   default=Path(__file__).with_name('surface_mixture_v7_audit.json'))
    a = p.parse_args()
    result = audit(a.n, a.length, a.seed, generator=a.generator,
                   section_rate=a.section_rate, detail_rate=a.detail_rate,
                   scope_question_rate=a.scope_question_rate,
                   quote_rate=a.quote_rate, source_rate=a.source_rate)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print({k: result[k] for k in ('labels', 'scope_answer_sites',
                                 'full_prefix_supported_alternative_keys',
                                 'details_after_intervening_narrative')})

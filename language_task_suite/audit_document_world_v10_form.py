"""Paired lexical × truth-conditional discourse-act V10 form screen."""

import argparse
import json
import random
from pathlib import Path

from audit_surface_repetition import WORD
from compare_document_mixture_v7_fineweb_form import panel
from compare_v12_fineweb_fresh import first_words
from document_world_v8_discourse import draw_varied as draw_v8, realize as realize_v8
from document_world_v9_typed_lexicon import draw_lexicon, lexicalize_document
from document_world_v10_fact_acts import plan_acts, realize as realize_v10


def paired(seed, n, words, source_rate, act_rate, exponent):
    rng = random.Random(seed)
    texts = {key: [] for key in ('v8', 'lexical', 'acts', 'both')}
    draws = act_count = 0
    while len(texts['v8']) < n:
        draws += 1
        base = draw_v8(rng, syntax_entropy=1., source_delay_rate=0.,
                       extra_source_recap_rate=source_rate)
        world = base.latent
        world['fact_act_plan'] = plan_acts(world, act_rate)
        world['surface_lexicon'] = draw_lexicon(
            random.Random(world['realization_seed'] ^ 0xC019), exponent)
        v8 = realize_v8(world, syntax_entropy=1.)
        acts = realize_v10(world, syntax_entropy=1.,
                           form_entropy=1.)
        docs = dict(v8=v8, lexical=lexicalize_document(v8, world),
                    acts=acts, both=lexicalize_document(acts, world))
        assert all([(s.family, s.answer) for s in doc.sites] ==
                   [(s.family, world['surface_lexicon'].get(s.answer,
                                                             s.answer))
                    for s in v8.sites]
                   for name, doc in docs.items() if name in ('lexical', 'both'))
        crops = {name: first_words(doc.text(), words)
                 for name, doc in docs.items()}
        if any(len(WORD.findall(text)) < words for text in crops.values()):
            continue
        act_count += len(world['fact_act_plan'])
        for name, text in crops.items():
            texts[name].append(text)
        assert draws <= 100*n
    return dict(draws=draws, acts_per_eligible_doc=act_count/n,
                panels={name: panel(rows, 40)
                        for name, rows in texts.items()})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=64)
    p.add_argument('--words', type=int, default=100)
    p.add_argument('--seed', type=int, default=85119)
    p.add_argument('--source-rate', type=float, default=0.)
    p.add_argument('--act-rate', type=float, default=.8)
    p.add_argument('--exponent', type=float, default=0.)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = dict(config=dict(n=a.n, words=a.words, seed=a.seed,
                              source_rate=a.source_rate,
                              act_rate=a.act_rate,
                              exponent=a.exponent),
                  panels=[paired(a.seed+i, a.n, a.words, a.source_rate,
                                 a.act_rate, a.exponent) for i in (0, 1)],
                  caveat='Same sampled worlds and eligibility across arms. '
                         'Four arms separate lexical expansion and extra '
                         'truth-conditional discourse acts. This does not '
                         'measure semantic coherence or oracle regret.')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    for i, row in enumerate(result['panels']):
        print('seed', a.seed+i, 'draws', row['draws'],
              'acts/doc', round(row['acts_per_eligible_doc'], 2))
        for name, p in row['panels'].items():
            print(name, 'types', p['word_types'], 'top10',
                  round(p['top10_share'], 3), 'repeat40',
                  round(p['repetition']['repeated_word_fraction'], 3))


if __name__ == '__main__':
    main()

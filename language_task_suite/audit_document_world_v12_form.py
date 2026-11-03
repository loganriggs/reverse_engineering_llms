"""Paired form screen for full versus sampled partial initial facts."""

import argparse
import json
import random
from pathlib import Path

from audit_surface_repetition import WORD
from compare_document_mixture_v7_fineweb_form import panel
from compare_v12_fineweb_fresh import first_words
from document_world_v10_fact_acts import draw_varied as draw_v10
from document_world_v11_initial_facts import plan_initial, realize as realize_v11
from document_world_v12_partial_facts import plan_partial


def paired(seed, n, words, source_rate, optional_rate):
    rng = random.Random(seed)
    texts = {k: [] for k in ('v8', 'v9', 'full_facts',
                             'full_facts_lexical', 'partial_facts',
                             'partial_facts_lexical')}
    first_site = {k: [] for k in texts}
    visible_facts = draws = 0
    while len(texts['v8']) < n:
        draws += 1
        base = draw_v10(rng, act_rate=0., lexical_exponent=0.,
                        aside_rate=0., context_rate=0.,
                        syntax_entropy=1., source_delay_rate=0.,
                        extra_source_recap_rate=source_rate)
        world = base.latent
        plans = {
            'full': plan_initial(world, 1., 1.),
            'partial': plan_partial(world, optional_rate, 1., 1.),
        }
        docs = {}
        for group, plan in plans.items():
            world['initial_fact_plan'] = plan
            docs[group + '_facts'] = realize_v11(
                world, lexical_enabled=False, act_enabled=False,
                aside_rate=0., context_rate=0., syntax_entropy=1.)
            docs[group + '_facts_lexical'] = realize_v11(
                world, lexical_enabled=True, act_enabled=False,
                aside_rate=0., context_rate=0., syntax_entropy=1.)
        docs['v8'] = realize_v11(world, initial_enabled=False,
                                act_enabled=False, lexical_enabled=False,
                                aside_rate=0., context_rate=0.,
                                syntax_entropy=1.)
        docs['v9'] = realize_v11(world, initial_enabled=False,
                                act_enabled=False, lexical_enabled=True,
                                aside_rate=0., context_rate=0.,
                                syntax_entropy=1.)
        crops = {name: first_words(doc.text(), words)
                 for name, doc in docs.items()}
        if any(len(WORD.findall(text)) < words for text in crops.values()):
            continue
        visible_facts += len(plans['partial'])
        for name, text in crops.items():
            texts[name].append(text)
            first_site[name].append(docs[name].sites[0].index)
        assert draws <= 100*n
    return dict(draws=draws, partial_facts_per_eligible_doc=visible_facts/n,
                panels={name: panel(rows, 40)
                        for name, rows in texts.items()},
                mean_first_site_token={name: sum(first_site[name])/n
                                       for name in texts})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=64)
    p.add_argument('--words', type=int, default=100)
    p.add_argument('--seed', type=int, default=92121)
    p.add_argument('--source-rate', type=float, default=0.)
    p.add_argument('--optional-rate', type=float, default=.35)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = dict(config=dict(n=a.n, words=a.words, seed=a.seed,
                              source_rate=a.source_rate,
                              optional_rate=a.optional_rate),
                  panels=[paired(a.seed+i, a.n, a.words, a.source_rate,
                                 a.optional_rate) for i in (0, 1)],
                  caveat='Paired generated worlds and first-100-word '
                         'eligibility. V12 disables incidental and extra '
                         'post-event acts to isolate initial fact sampling. '
                         'Natural-text benchmarks are separate panels.')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    for i, row in enumerate(result['panels']):
        print('seed', a.seed+i, 'draws', row['draws'],
              'facts/doc', round(row['partial_facts_per_eligible_doc'], 2))
        for name, p in row['panels'].items():
            print(name, p['word_types'],
                  round(p['top10_share'], 3),
                  round(p['repetition']['repeated_word_fraction'], 3),
                  round(row['mean_first_site_token'][name], 1))


if __name__ == '__main__':
    main()

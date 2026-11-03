"""Matched V8–V11 factorial form screen: lexical, acts, initial-fact order."""

import argparse
import json
import random
from pathlib import Path

from audit_surface_repetition import WORD
from compare_document_mixture_v7_fineweb_form import panel
from compare_v12_fineweb_fresh import first_words
from document_world_v10_fact_acts import draw_varied as draw_v10
from document_world_v11_initial_facts import plan_initial, realize


ARMS = {
    'v8': (False, False, False),
    'lexical': (False, False, True),
    'acts': (False, True, False),
    'acts_lexical': (False, True, True),
    'initial': (True, False, False),
    'initial_lexical': (True, False, True),
    'initial_acts': (True, True, False),
    'initial_acts_lexical': (True, True, True),
}


def paired(seed, n, words, source_rate):
    rng = random.Random(seed)
    texts = {key: [] for key in ARMS}
    first_site = {key: [] for key in ARMS}
    draws = 0
    while len(texts['v8']) < n:
        draws += 1
        base = draw_v10(rng, act_rate=.8, lexical_exponent=0.,
                        syntax_entropy=1., source_delay_rate=0.,
                        extra_source_recap_rate=source_rate)
        world = base.latent
        world['initial_fact_plan'] = plan_initial(world, 1., 1.)
        docs = {name: realize(world, initial_enabled=initial,
                              act_enabled=acts, lexical_enabled=lexical,
                              syntax_entropy=1.)
                for name, (initial, acts, lexical) in ARMS.items()}
        assert all([s.family for s in doc.sites] ==
                   [s.family for s in docs['v8'].sites]
                   for doc in docs.values())
        crops = {name: first_words(doc.text(), words)
                 for name, doc in docs.items()}
        if any(len(WORD.findall(text)) < words for text in crops.values()):
            continue
        for name, text in crops.items():
            texts[name].append(text)
            first_site[name].append(docs[name].sites[0].index)
        assert draws <= 100*n
    return dict(draws=draws,
                panels={name: panel(rows, 40)
                        for name, rows in texts.items()},
                mean_first_site_token={name: sum(first_site[name])/n
                                       for name in ARMS})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=64)
    p.add_argument('--words', type=int, default=100)
    p.add_argument('--seed', type=int, default=89121)
    p.add_argument('--source-rate', type=float, default=0.)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = dict(config=dict(n=a.n, words=a.words, seed=a.seed,
                              source_rate=a.source_rate, act_rate=.8,
                              lexical_exponent=0.,
                              initial_order_entropy=1.,
                              initial_form_entropy=1.),
                  panels=[paired(a.seed+i, a.n, a.words, a.source_rate)
                          for i in (0, 1)],
                  caveat='Eight paired generated arms share worlds and '
                         'first-100-word eligibility. No natural text '
                         'bodies are stored; compare saved independent '
                         'FineWeb panels as scale references.')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    for i, row in enumerate(result['panels']):
        print('seed', a.seed+i, 'draws', row['draws'])
        for name, p in row['panels'].items():
            print(name, p['word_types'],
                  round(p['top10_share'], 3),
                  round(p['repetition']['repeated_word_fraction'], 3),
                  round(row['mean_first_site_token'][name], 1))


if __name__ == '__main__':
    main()

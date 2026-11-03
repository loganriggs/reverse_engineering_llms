"""First-40-word V12 fact-budget sweep on paired sampled worlds."""

import argparse
import json
import random
from pathlib import Path

from audit_surface_repetition import WORD
from compare_document_mixture_v7_fineweb_form import panel
from document_world_v10_fact_acts import draw_varied as draw_v10
from document_world_v12_partial_facts import plan_partial, realize


RATES = (0., .2, .35, .7, 1.)


def run(seed, n):
    rng = random.Random(seed)
    texts = {str(rate): [] for rate in RATES}
    texts['v8'] = []
    fact_counts = {str(rate): 0 for rate in RATES}
    for _ in range(n):
        world = draw_v10(
            rng, act_rate=0., lexical_exponent=0., aside_rate=0.,
            context_rate=0., syntax_entropy=1., source_delay_rate=0.,
            extra_source_recap_rate=.5).latent
        base = realize(world, lexical_enabled=True,
                       initial_enabled=False, syntax_entropy=1.)
        texts['v8'].append(' '.join(WORD.findall(base.text())[:40]))
        for rate in RATES:
            key = str(rate)
            world['initial_fact_plan'] = plan_partial(world, rate, 1., 1.)
            doc = realize(world, lexical_enabled=True, syntax_entropy=1.)
            fact_counts[key] += len(world['initial_fact_plan'])
            texts[key].append(' '.join(WORD.findall(doc.text())[:40]))
    return dict(seed=seed, n=n,
                mean_visible_facts={key: value/n
                                    for key, value in fact_counts.items()},
                first40_panels={key: panel(rows, 40)
                                for key, rows in texts.items()})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=128)
    p.add_argument('--seed', type=int, default=93221)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = dict(config=dict(n=a.n, seed=a.seed,
                              source_rate=.5, lexical_exponent=0.,
                              aside_rate=0., context_rate=0.),
                  panels=[run(a.seed+i, a.n) for i in (0, 1)],
                  caveat='All arms use identical worlds. Top-ten share and '
                         'types are over cropped first-40-word windows, '
                         'not directly comparable to prior first-100-word '
                         'FineWeb panels.')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    for row in result['panels']:
        for key, panel_ in row['first40_panels'].items():
            print(row['seed'], key,
                  round(row['mean_visible_facts'].get(key, 0), 2),
                  round(panel_['top10_share'], 3),
                  round(panel_['repetition']['repeated_word_fraction'], 3),
                  panel_['word_types'])


if __name__ == '__main__':
    main()

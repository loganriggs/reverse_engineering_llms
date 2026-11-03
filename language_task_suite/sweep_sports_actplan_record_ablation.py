"""Matched-world form ablation for optional pre/post record acts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

from audit_document_structure_vs_fineweb import structure
from compare_v12_fineweb_fresh import first_words
from sports_document_actplan_v1 import realize, sample_plan
from sports_document_dgp import realize as base_realize


ARMS = {
    'no_records': {'record_rate': 0.},
    'pregame_only': {'record_rate': 1., 'post_record_rate': 0.},
    'pre_and_post': {'record_rate': 1., 'post_record_rate': 1.},
    'default': {},
}


def panel(seed, n, words, source, group):
    rng = random.Random(seed)
    crops = {key: [] for key in ARMS | {'fixed_empirical': {}}}
    draws = 0
    while len(crops['default']) < n:
        draws += 1
        if draws > 200*n:
            raise RuntimeError('Too few shared word crops')
        plan = sample_plan(rng, sbnation_rate=int(source == 'sbnation'),
                           source_group=group)
        texts = {name: first_words(realize(plan, **kwargs).text, words)
                 for name, kwargs in ARMS.items()}
        texts['fixed_empirical'] = first_words(base_realize(
            plan.base, pair_rate=0., stat_richness=1.,
            discourse_entropy=1., lead_stat_rate=.25,
            empirical_stat_rate=1.).text, words)
        if min(len(text.split()) for text in texts.values()) < words:
            continue
        for name, text in texts.items():
            crops[name].append(text)
    return {'draws': draws,
            'form': {name: structure(texts)
                     for name, texts in crops.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=38201)
    parser.add_argument('--n', type=int, default=64)
    parser.add_argument('--words', type=int, default=55)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = {'config': {'seed': args.seed, 'n': args.n,
                      'words': args.words}, 'arms': {},
           'caveat': 'All arms use identical sampled worlds and are jointly '
                     'eligible for the word crop; this isolates record-act '
                     'surface effects, not overall linguistic realism.'}
    for source in ('rotowire', 'sbnation'):
        for group in ('train', 'heldout'):
            key = source+'_'+group
            out['arms'][key] = panel(args.seed+len(out['arms'])*100,
                                     args.n, args.words, source, group)
            print(key, out['arms'][key]['draws'],
                  {name: round(v['repeated_across_docs_trigram_occurrence_share'], 3)
                   for name, v in out['arms'][key]['form'].items()},
                  flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2)+'\n')


if __name__ == '__main__':
    main()

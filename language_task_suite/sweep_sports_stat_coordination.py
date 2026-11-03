"""Matched-world form and answer audit for coordinated player-stat clauses."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random

from audit_document_structure_vs_fineweb import structure
from compare_v12_fineweb_fresh import first_words
from sports_document_actplan_v1 import (
    answer_from_visible_prefix, realize, sample_plan,
)


ARMS = {
    'empirical_singletons': dict(empirical_stat_rate=1., pair_rate=0.,
                                 record_rate=0.),
    'mixed_coordination': dict(empirical_stat_rate=.5, pair_rate=1.,
                               record_rate=0.),
    'fixed_coordination': dict(empirical_stat_rate=0., pair_rate=1.,
                               record_rate=0.),
}


def panel(seed, n, words, source, group):
    rng = random.Random(seed)
    crops = {name: [] for name in ARMS}
    counts = Counter()
    draws = 0
    while len(crops['empirical_singletons']) < n:
        draws += 1
        if draws > n*200:
            raise RuntimeError('Too few jointly eligible documents')
        plan = sample_plan(rng, sbnation_rate=int(source == 'sbnation'),
                           source_group=group)
        documents = {name: realize(plan, **kwargs)
                     for name, kwargs in ARMS.items()}
        texts = {name: first_words(doc.text, words)
                 for name, doc in documents.items()}
        if min(len(text.split()) for text in texts.values()) < words:
            continue
        for name, doc in documents.items():
            for site in doc.sites:
                assert answer_from_visible_prefix(
                    doc.text[:site.start], site.kind) == site.answer
                counts[name+'/sites_checked'] += 1
            crops[name].append(texts[name])
    return {'draws': draws, 'sites_checked': dict(counts),
            'form': {name: structure(texts)
                     for name, texts in crops.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=40601)
    parser.add_argument('--n', type=int, default=64)
    parser.add_argument('--words', type=int, default=55)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = {'config': vars(args) | {'output': str(args.output)},
           'caveat': 'Matched generated worlds and jointly eligible word '
                     'crops. This tests a short coordinated-stat grammar; '
                     'it is not a realism pass. No human text is saved.',
           'arms': {}}
    for source in ('rotowire', 'sbnation'):
        for group in ('train', 'heldout'):
            key = source+'_'+group
            result = panel(args.seed+100*len(out['arms']), args.n,
                           args.words, source, group)
            out['arms'][key] = result
            print(key, result['draws'], {
                name: round(value[
                    'repeated_across_docs_trigram_occurrence_share'], 3)
                for name, value in result['form'].items()}, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2)+'\n')


if __name__ == '__main__':
    main()

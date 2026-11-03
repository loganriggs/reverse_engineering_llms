"""Semantic and format audit of short event-derived live-score ledgers."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import statistics

from audit_document_structure_vs_fineweb import structure
from compare_v12_fineweb_fresh import first_words
from sports_live_ledger_dgp import (answer_from_visible_prefix,
                                    ledger_pool, sample_document)


def panel(seed, n, source, group):
    counts = Counter()
    described, bare = [], []
    lengths = {'described': [], 'bare': []}
    examples = []
    for i in range(n):
        first = random.Random(seed+i)
        second = random.Random(seed+i)
        with_descriptions = sample_document(first, source, group, True)
        without_descriptions = sample_document(second, source, group, False)
        for doc in (with_descriptions, without_descriptions):
            assert doc.latent['source'] == source
            assert doc.latent['split'] == group
            assert len(doc.sites) == 3
            for site in doc.sites:
                assert doc.tokens[site.index] == site.target
                kind = 'copy' if site.family == 'live_score_copy' else 'run'
                assert answer_from_visible_prefix(doc.tokens[:site.index],
                                                  kind) \
                    == site.target
                counts['visible_answer_sites_checked'] += 1
        assert with_descriptions.latent['game_id'] == \
            without_descriptions.latent['game_id']
        assert with_descriptions.latent['start_score'] == \
            without_descriptions.latent['start_score']
        assert with_descriptions.latent['end_score'] == \
            without_descriptions.latent['end_score']
        assert tuple(site.target for site in with_descriptions.sites) == \
            tuple(site.target for site in without_descriptions.sites)
        counts['paired_worlds_checked'] += 1
        counts['actions_sum'] += with_descriptions.latent['actions']
        counts['period/'+str(with_descriptions.latent['period'])] += 1
        for key, doc, target in (('described', with_descriptions, described),
                                 ('bare', without_descriptions, bare)):
            lengths[key].append(len(doc.tokens))
            target.append(first_words(doc.text(), 80))
        if len(examples) < 2:
            # Structured-only examples contain no raw source description.
            examples.append(without_descriptions.text())
    return {'counts': dict(counts),
            'mean_tokens': {name: round(statistics.mean(values), 1)
                            for name, values in lengths.items()},
            'first80_structure': {'described': structure(described),
                                  'bare': structure(bare)},
            'structured_examples': examples}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=10306)
    parser.add_argument('--n', type=int, default=200)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    pools = ledger_pool()
    for source in ('rotowire', 'sbnation'):
        train = {entry[0][0] for entry in pools[source, 'train']}
        heldout = {entry[0][0] for entry in pools[source, 'heldout']}
        assert not train & heldout
    out = {'config': {'seed': args.seed, 'n_per_arm': args.n},
           'eligible_distinct_games': {source+'_'+group: len(
               pools[source, group]) for source in ('rotowire', 'sbnation')
                                     for group in ('train', 'heldout')},
           'arms': {},
           'caveat': 'Original play descriptions are empirical text, not '
                     'a symbolic all-token oracle. The live-score wrapper '
                     'is a controlled genre pilot, not a natural-report '
                     'realism pass. Only structured-only generated examples '
                     'and aggregate metrics are saved.'}
    for source in ('rotowire', 'sbnation'):
        for group in ('train', 'heldout'):
            key = source+'_'+group
            result = panel(args.seed+10000*len(out['arms']), args.n,
                           source, group)
            out['arms'][key] = result
            print(key, result['counts'], result['mean_tokens'], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2)+'\n')


if __name__ == '__main__':
    main()

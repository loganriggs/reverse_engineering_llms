"""Paired V1/V2 or V2/V3 form comparison on identical sports worlds."""

import argparse
import json
import random
from pathlib import Path

from audit_document_structure_vs_fineweb import structure
from audit_sports_document_dgp import _window
from compare_v12_fineweb_fresh import first_words
from sports_document_dgp import realize, sample_plan


def panel(seed, n, words, group, comparison='discourse'):
    rng = random.Random(seed)
    versions = ('v1', 'v2') if comparison == 'discourse' else ('v2', 'v3')
    docs = {version: [] for version in versions}
    windows = {version: {kind: [] for kind in
                         ('winner', 'margin', 'top_player')}
               for version in docs}
    examples = []
    draws = 0
    while len(docs[versions[0]]) < n:
        draws += 1
        plan = sample_plan(rng, sbnation_rate=.5,
                           source_group=group)
        if comparison == 'discourse':
            old = realize(plan, form_entropy=1., pair_rate=1.)
            new = realize(plan, form_entropy=1., pair_rate=1.,
                          discourse_entropy=1., lead_stat_rate=.5)
        else:
            old = realize(plan, form_entropy=1., pair_rate=1.,
                          discourse_entropy=1., lead_stat_rate=.5)
            new = realize(plan, form_entropy=1., pair_rate=1.,
                          discourse_entropy=1., lead_stat_rate=.5,
                          stat_richness=1.)
        crops = {version: first_words(doc.text, words)
                 for version, doc in zip(versions, (old, new))}
        if any(len(crop.split()) < words for crop in crops.values()):
            continue
        for version, doc in zip(versions, (old, new)):
            docs[version].append(crops[version])
            for site in doc.sites:
                windows[version][site.kind].append(_window(doc, site))
        if len(examples) < 2:
            examples.append({versions[0]: old.text, versions[1]: new.text})
    return {'draws': draws,
            'form': {v: structure(rows) for v, rows in docs.items()},
            'site_windows': {v: {k: structure(rows)
                                 for k, rows in kinds.items()}
                             for v, kinds in windows.items()},
            'examples': examples}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=64)
    p.add_argument('--words', type=int, default=55)
    p.add_argument('--seed', type=int, default=90311)
    p.add_argument('--comparison', choices=('discourse', 'stats'),
                   default='discourse')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    output = {'config': {'n': a.n, 'words': a.words, 'seed': a.seed,
                         'comparison': a.comparison},
              'train': panel(a.seed, a.n, a.words, 'train', a.comparison),
              'heldout': panel(a.seed+1, a.n, a.words, 'heldout',
                               a.comparison)}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(output, indent=2)+'\n')
    versions = ('v1', 'v2') if a.comparison == 'discourse' else ('v2', 'v3')
    for group in ('train', 'heldout'):
        for version in versions:
            result = output[group]['form'][version]
            print(group, version, result['word_types'],
                  result['repeated_across_docs_trigram_occurrence_share'])


if __name__ == '__main__':
    main()

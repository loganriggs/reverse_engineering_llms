"""Paired V8 document-introduction planning screen, without natural-text fetches.

Natural references are the already saved, independently sampled FineWeb form
panels. No source text or sampled documents are stored in this result.
"""

import argparse
import json
import random
from pathlib import Path

from audit_surface_repetition import WORD
from compare_document_mixture_v7_fineweb_form import panel
from compare_v12_fineweb_fresh import first_words
from document_world_v8_discourse import draw_varied, realize


def paired_panel(seed, n, words, source_rate):
    rng = random.Random(seed)
    rows = {0: [], 1: []}
    first_site = {0: [], 1: []}
    changed = draws = 0
    while len(rows[0]) < n:
        draws += 1
        base = draw_varied(rng, syntax_entropy=1.,
                           extra_source_recap_rate=source_rate,
                           source_delay_rate=0.)
        a = realize(base.latent, syntax_entropy=1.,
                    intro_interleave_rate=0.)
        b = realize(base.latent, syntax_entropy=1.,
                    intro_interleave_rate=1.)
        assert [(s.family, s.answer) for s in a.sites] == [
            (s.family, s.answer) for s in b.sites]
        crops = [first_words(doc.text(), words) for doc in (a, b)]
        if any(len(WORD.findall(crop)) < words for crop in crops):
            continue
        changed += a.tokens != b.tokens
        for key, (doc, crop) in enumerate(zip((a, b), crops)):
            rows[key].append(crop)
            first_site[key].append(doc.sites[0].index)
        assert draws <= 100*n
    return dict(
        draws=draws, changed_documents=changed,
        panels={str(key): panel(rows[key], 40) for key in rows},
        mean_first_site_token={str(key): sum(first_site[key])/n
                               for key in rows},
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=64)
    p.add_argument('--words', type=int, default=100)
    p.add_argument('--seed', type=int, default=73381)
    p.add_argument('--source-rate', type=float, default=0.)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = dict(config=dict(n=a.n, words=a.words, seed=a.seed,
                              source_rate=a.source_rate),
                  panels=[paired_panel(a.seed+i, a.n, a.words,
                                       a.source_rate) for i in (0, 1)],
                  caveat='Paired generated worlds; first-100-word eligible '
                         'documents only. Intro timing varies but lexicon '
                         'and fact templates do not. Compare to saved '
                         'FineWeb panels separately; no natural-text fetch.')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    for i, row in enumerate(result['panels']):
        print('seed', a.seed+i, 'changed', row['changed_documents'],
              'draws', row['draws'])
        for key, p in row['panels'].items():
            print('intro', key, 'types', p['word_types'], 'top10',
                  round(p['top10_share'], 3), 'repeat40',
                  round(p['repetition']['repeated_word_fraction'], 3),
                  'first_site', round(row['mean_first_site_token'][key], 1))


if __name__ == '__main__':
    main()

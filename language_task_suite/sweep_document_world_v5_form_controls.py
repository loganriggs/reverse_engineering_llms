"""Controlled V5 renderer sweep with identical underlying sampled worlds."""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from audit_surface_repetition import WORD, form_cues, score
from compare_v12_fineweb_fresh import first_words
from document_world_v5_clause_realizations import (
    answer_from_visible_prefix, draw_varied)


BASE = dict(style_entropy=.7, aside_rate=.3, paragraph_rate=.22,
            header_rate=.5, source_delay_rate=1.)
SETTINGS = [('base', {})]
for parameter, values in (
        ('style_entropy', (0., .35, 1.)),
        ('aside_rate', (0., .6, 1.)),
        ('paragraph_rate', (0., .5, 1.)),
        ('header_rate', (0., 1.))):
    SETTINGS.extend((f'{parameter}={value:g}', {parameter: value})
                    for value in values)


def panel(seed, n, settings, words=100):
    rng = random.Random(seed)
    texts = []
    worlds = []
    sites = 0
    for _ in range(n):
        doc = draw_varied(rng, **settings)
        worlds.append(doc.latent)
        sites += len(doc.sites)
        for site in doc.sites:
            assert answer_from_visible_prefix(
                doc.tokens[:site.index]) == site.answer
        cropped = first_words(doc.text(), words)
        if len(WORD.findall(cropped)) >= words:
            texts.append(cropped)
    assert texts
    tokens = [word.lower() for text in texts for word in WORD.findall(text)]
    counts = Counter(tokens)
    return (dict(n_documents=n, n_eligible=len(texts), n_answer_sites=sites,
                 word_types=len(counts),
                 top10_share=sum(n for _, n in counts.most_common(10)) /
                 len(tokens),
                 repeated_word_fraction=score(texts, 40)[
                     'repeated_word_fraction'],
                 form=form_cues(texts)), worlds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=300)
    parser.add_argument('--seeds', default='7723,8823')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = {'config': {'n': args.n, 'seeds': args.seeds,
                         'base': BASE, 'first_words': 100}, 'panels': {}}
    for seed in map(int, args.seeds.split(',')):
        reference_worlds = None
        for name, change in SETTINGS:
            stats, worlds = panel(seed, args.n, BASE | change)
            if reference_worlds is None:
                reference_worlds = worlds
            else:
                assert worlds == reference_worlds, 'Latent worlds changed'
            result['panels'][f'{seed}:{name}'] = stats
            print(seed, name, stats['n_eligible'], stats['word_types'],
                  round(stats['top10_share'], 3),
                  round(stats['repeated_word_fraction'], 3), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()

"""Paired lexical-axis form screen for V8 worlds and V9 realizations.

Uses previously frozen FineWeb aggregate panels as scale references; this
script does not fetch natural text or save generated document bodies.
"""

import argparse
import json
import random
from pathlib import Path

from audit_surface_repetition import WORD
from compare_document_mixture_v7_fineweb_form import panel
from compare_v12_fineweb_fresh import first_words
from document_world_v8_discourse import draw_varied as draw_v8, realize as realize_v8
from document_world_v9_typed_lexicon import draw_lexicon, realize as realize_v9


def paired(seed, n, words, extra_source_rate):
    rng = random.Random(seed)
    texts = {'v8': [], 'v9_uniform': [], 'v9_zipf': []}
    draws = 0
    while len(texts['v8']) < n:
        draws += 1
        base = draw_v8(rng, syntax_entropy=1., source_delay_rate=0.,
                       extra_source_recap_rate=extra_source_rate)
        world = base.latent
        rendered = {'v8': realize_v8(world, syntax_entropy=1.)}
        for name, exponent in (('v9_uniform', 0.), ('v9_zipf', 1.)):
            revised = dict(world)
            revised['surface_lexicon'] = draw_lexicon(
                random.Random(world['realization_seed'] +
                              (101 if exponent == 0. else 103)), exponent)
            rendered[name] = realize_v9(revised, syntax_entropy=1.)
            assert [(s.family, s.answer == rendered[name].latent[
                'surface_lexicon'].get(old.answer, old.answer))
                    for s, old in zip(rendered[name].sites,
                                      rendered['v8'].sites, strict=True)] == [
                (s.family, True) for s in rendered[name].sites]
        crops = {name: first_words(doc.text(), words)
                 for name, doc in rendered.items()}
        if any(len(WORD.findall(text)) < words for text in crops.values()):
            continue
        for name, text in crops.items():
            texts[name].append(text)
        assert draws <= 100*n
    return dict(draws=draws,
                panels={name: panel(rows, 40)
                        for name, rows in texts.items()})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=64)
    p.add_argument('--words', type=int, default=100)
    p.add_argument('--seed', type=int, default=73981)
    p.add_argument('--extra-source-rate', type=float, default=0.)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = dict(config=dict(n=a.n, words=a.words, seed=a.seed,
                              extra_source_rate=a.extra_source_rate),
                  panels=[paired(a.seed+i, a.n, a.words,
                                 a.extra_source_rate) for i in (0, 1)],
                  caveat='Same world, syntax seed, and eligibility across '
                         'generated arms; lexical maps differ. Distinct '
                         'natural reference panels are saved elsewhere. '
                         'No text bodies are saved here.')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    for i, row in enumerate(result['panels']):
        print('seed', a.seed+i, 'draws', row['draws'])
        for name, p in row['panels'].items():
            print(name, 'types', p['word_types'], 'top10',
                  round(p['top10_share'], 3), 'repeat40',
                  round(p['repetition']['repeated_word_fraction'], 3))


if __name__ == '__main__':
    main()

"""Compare V14 clause-frame diversity to a fresh FineWeb offset range."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path

from audit_document_structure_vs_fineweb import natural, structure
from compare_v12_fineweb_fresh import first_words
from surface_mixture_v6 import render
from surface_mixture_v14_conditional import SurfaceMixtureV14Conditional
from train_v13_attributes import COMMON, REFERENCE


def sample_v14(n, seed, words, tokens):
    gen = SurfaceMixtureV14Conditional(
        seed=seed, **COMMON, **REFERENCE,
        conditional_continuation_rate=.35)
    rows = []
    sites = Counter()
    draws = 0
    while len(rows) < n:
        draws += 1
        ids, labels, _ = gen.sample_sequence(tokens)
        text = render([gen.id2token[index] for index in ids])
        crop = first_words(text, words)
        if len(crop.split()) < words:
            continue
        rows.append(crop)
        sites.update(label for label in labels if label != 'frame')
    return {'draws': draws, 'form': structure(rows),
            'sampled_site_counts': dict(sites)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--skip', type=int, default=94500)
    p.add_argument('--n', type=int, default=128)
    p.add_argument('--words', type=int, default=100)
    p.add_argument('--tokens', type=int, default=256)
    p.add_argument('--seed', type=int, default=52301)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    reference, scanned = natural(a.skip, a.n, a.words)
    first = sample_v14(a.n, a.seed, a.words, a.tokens)
    second = sample_v14(a.n, a.seed+1, a.words, a.tokens)
    result = {'config': {'skip': a.skip, 'n': a.n, 'words': a.words,
                         'toy_tokens': a.tokens, 'toy_seeds': [a.seed,
                                                              a.seed+1]},
              'fineweb_rows_scanned': scanned,
              'panels': {'fineweb': reference, 'v14_seed_a': first,
                         'v14_seed_b': second},
              'caveat': 'First-100-word descriptive form screen only. '
                        'FineWeb is independent web text; regex sentence '
                        'segmentation treats headings and fragments as '
                        'sentence starts. No source passages are saved.'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    for name, row in result['panels'].items():
        form = row if name == 'fineweb' else row['form']
        print(name, 'word types', form['word_types'],
              'first starts', form['first_sentence_start_types'],
              'sentence starts', form['sentence_start_types'],
              'top10', round(form['top10_word_share'], 3),
              'trigram repeat', round(form[
                  'repeated_across_docs_trigram_occurrence_share'], 3),
              flush=True)
    os._exit(0)  # datasets/pyarrow streaming may hang on interpreter exit


if __name__ == '__main__':
    main()

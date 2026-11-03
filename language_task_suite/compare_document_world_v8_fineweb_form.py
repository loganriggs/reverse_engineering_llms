"""Matched-world V5/V8 and independent FineWeb first-100-word form panels."""

import argparse
import gc
import json
import os
import random
import sys
from itertools import islice
from pathlib import Path

from datasets import load_dataset

from audit_surface_repetition import WORD
from compare_document_mixture_v7_fineweb_form import panel
from compare_v12_fineweb_fresh import first_words
from document_world_v5_clause_realizations import draw_varied as draw_v5
from document_world_v8_discourse import draw_varied as draw_v8
from document_world_v8_discourse import realize as realize_v8


def matched_generated(n, seed, words):
    rng5, rng8 = random.Random(seed), random.Random(seed)
    v5, v8, v8_no_context = [], [], []
    draws = 0
    while len(v5) < n and draws < n * 30:
        draws += 1
        left = draw_v5(rng5, source_delay_rate=1.)
        right = draw_v8(rng8, source_delay_rate=1.)
        assert left.latent == right.latent
        assert [(site.family, site.answer) for site in left.sites] == [
            (site.family, site.answer) for site in right.sites]
        text5 = first_words(left.text(), words)
        text8 = first_words(right.text(), words)
        text8_no_context = first_words(
            realize_v8(right.latent, context_rate=0.).text(), words)
        if (len(WORD.findall(text5)) >= words and
                len(WORD.findall(text8)) >= words and
                len(WORD.findall(text8_no_context)) >= words):
            v5.append(text5)
            v8.append(text8)
            v8_no_context.append(text8_no_context)
    assert len(v5) == n
    return v5, v8_no_context, v8, draws


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip', type=int, default=36000)
    parser.add_argument('--n', type=int, default=64)
    parser.add_argument('--words', type=int, default=100)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    left, no_context, right, draws = matched_generated(
        args.n, seed=123 + args.skip, words=args.words)
    dataset = load_dataset('HuggingFaceFW/fineweb', name='sample-10BT',
                           split='train', streaming=True)
    stream = iter(dataset)
    try:
        selected = list(islice(stream, args.skip, args.skip + args.n * 3))
    finally:
        if hasattr(stream, 'close'):
            stream.close()
    del stream, dataset
    gc.collect()
    natural = []
    for row in selected:
        text = first_words(row['text'], args.words)
        if len(WORD.findall(text)) >= args.words:
            natural.append(text)
        if len(natural) == args.n:
            break
    assert len(natural) == args.n
    panels = {name: panel(texts, 40) for name, texts in (
        ('v5', left), ('v8_no_context', no_context),
        ('v8', right), ('fineweb', natural))}
    result = dict(config=dict(skip=args.skip, n=args.n, words=args.words,
                              generator_seed=123 + args.skip,
                              generated_draws=draws,
                              dataset='HuggingFaceFW/fineweb sample-10BT train'),
                  panels=panels,
                  caveat='V5/V8 share each latent world and answer sequence; '
                         'natural windows are separate. Form screen only, '
                         'not semantic realism. No source text saved.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    for name, row in panels.items():
        print(name, row['word_types'], round(row['top10_share'], 3),
              round(row['repetition']['repeated_word_fraction'], 3),
              flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()

"""Independent form panels for V8 syntax, source prevalence, and delay knobs."""

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
from document_world_v8_discourse import draw_varied


def generated(n, seed, words, syntax, extra, delay):
    rng = random.Random(seed)
    texts, docs, source_sites = [], 0, 0
    while len(texts) < n:
        docs += 1
        doc = draw_varied(rng, syntax_entropy=syntax,
                          extra_source_recap_rate=extra,
                          source_delay_rate=delay)
        cropped = first_words(doc.text(), words)
        if len(WORD.findall(cropped)) >= words:
            texts.append(cropped)
            source_sites += sum(site.family == 'source_attribution'
                                for site in doc.sites)
        assert docs <= 100*n
    return texts, docs, source_sites


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip', type=int, default=38000)
    parser.add_argument('--n', type=int, default=64)
    parser.add_argument('--words', type=int, default=100)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    panels = {}
    for syntax in (0., 1.):
        for extra in (0., .5):
            for delay in (0., .5):
                key = f'syntax{syntax:g}_source{extra:g}_delay{delay:g}'
                texts, draws, source_sites = generated(
                    args.n, seed=34801+args.skip,
                    words=args.words, syntax=syntax, extra=extra,
                    delay=delay)
                panels[key] = panel(texts, 40)
                panels[key]['generated_draws'] = draws
                panels[key]['source_sites_per_eligible_doc'] = (
                    source_sites/args.n)
    dataset = load_dataset('HuggingFaceFW/fineweb', name='sample-10BT',
                           split='train', streaming=True)
    stream = iter(dataset)
    try:
        selected = list(islice(stream, args.skip, args.skip+args.n*3))
    finally:
        if hasattr(stream, 'close'):
            stream.close()
    del stream, dataset
    gc.collect()
    natural = []
    for row in selected:
        cropped = first_words(row['text'], args.words)
        if len(WORD.findall(cropped)) >= args.words:
            natural.append(cropped)
        if len(natural) == args.n:
            break
    assert len(natural) == args.n
    panels['fineweb'] = panel(natural, 40)
    output = dict(config=dict(skip=args.skip, n=args.n, words=args.words,
                              generated_seed=34801+args.skip,
                              source='HuggingFaceFW/fineweb sample-10BT train'),
                  panels=panels,
                  caveat='Reference source prevalence is extra rate 0; '
                         'extra .5 is an intentionally task-rich panel. '
                         'Each setting has its own eligibility sample; '
                         'the panels are descriptive, not matched worlds '
                         'or proof of semantic realism. No source text saved.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    for key, row in panels.items():
        print(key, row['word_types'], round(row['top10_share'], 3),
              round(row['repetition']['repeated_word_fraction'], 3),
              flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()

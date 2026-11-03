"""Independent form screen for V17 frequency sweeps; stores aggregates only."""

import argparse
import gc
import json
import os
import sys
from collections import Counter
from itertools import islice
from pathlib import Path

from datasets import load_dataset

from audit_surface_repetition import WORD, form_cues, score
from compare_v12_fineweb_fresh import first_words
from surface_mixture_v2 import SurfaceMixtureV2, render as render_v2
from surface_mixture_v17_typed_lexicon import SurfaceMixtureV17, render as render_v17


def panel(texts):
    words = [word.lower() for text in texts for word in WORD.findall(text)]
    frequency = Counter(words)
    return dict(word_types=len(frequency),
                hapax_types=sum(n == 1 for n in frequency.values()),
                top10_share=sum(n for _, n in frequency.most_common(10))/len(words),
                repetition=score(texts, 40), form=form_cues(texts))


def generated(factory, renderer, n, words):
    generator = factory()
    output = []
    for _ in range(n):
        ids, _, _ = generator.sample_sequence(words * 2)
        text = first_words(renderer(ids), words)
        assert len(WORD.findall(text)) >= words
        output.append(text)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip', type=int, default=34000)
    parser.add_argument('--n', type=int, default=64)
    parser.add_argument('--words', type=int, default=100)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    seed = 41 + args.skip
    panels = {'v2': panel(generated(lambda: SurfaceMixtureV2(seed=seed),
                                    render_v2, args.n, args.words))}
    for exponent in (0., .5, 1.):
        name = f'v17_exponent_{exponent:g}'
        panels[name] = panel(generated(
            lambda exponent=exponent: SurfaceMixtureV17(
                seed=seed, frequency_exponent=exponent),
            render_v17, args.n, args.words))
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
    panels['fineweb'] = panel(natural)
    result = dict(config=dict(skip=args.skip, n=args.n, words=args.words,
                              generator_seed=seed,
                              source='HuggingFaceFW/fineweb sample-10BT train',
                              lexical_prior_fit_offset=32000,
                              lexical_prior_fit_n=512),
                  panels=panels,
                  caveat='First-100-word form screen only. Frequency prior fit '
                         'on disjoint offsets. No source passages saved.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    for name, value in panels.items():
        print(name, value['word_types'], round(value['top10_share'], 3),
              round(value['repetition']['repeated_word_fraction'], 3),
              flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()

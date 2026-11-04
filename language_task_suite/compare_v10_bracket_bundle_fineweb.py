"""Fresh-source form screen for bracket-mediated V10 event coupling."""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import islice
import json
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from audit_surface_repetition import WORD, form_cues, score
from surface_mixture_v6 import render
from surface_mixture_v10_linked_events import SurfaceMixtureV10LinkedEvents
from surface_mixture_v10_bundled import SurfaceMixtureV10Bundled


def first_words(text, n):
    return [word.lower() for word in WORD.findall(text)[:n]]


def lexical(texts, n):
    words = [word for text in texts for word in first_words(text, n)]
    counts = Counter(words)
    return {'words': len(words), 'types': len(counts),
            'top_ten_fraction': sum(count for _, count in
                                    counts.most_common(10))/len(words)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--start', type=int, default=430000)
    p.add_argument('--n', type=int, default=192)
    p.add_argument('--length', type=int, default=256)
    p.add_argument('--seed', type=int, default=99111)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    dataset = load_dataset('HuggingFaceFW/fineweb', name='sample-10BT',
                           split='train', streaming=True)
    stream = iter(dataset)
    try:
        sources = [datum['text'] for datum in islice(stream, args.start,
                                                     args.start + args.n)]
    finally:
        if hasattr(stream, 'close'):
            stream.close()
    assert len(sources) == args.n
    natural = [tokenizer.decode(tokenizer.encode(text,
                 add_special_tokens=False).ids[:args.length],
                 skip_special_tokens=False) for text in sources]
    kwargs = dict(counterbalanced_queries=True, question_rate=.05,
                  count_rate=0., quote_rate=.08, source_rate=.15,
                  section_rate=.06, detail_rate=.8, scope_question_rate=.08,
                  source_object_mass=.15, source_subject_mass=.15,
                  bracket_rate=.06, linked_subject_mass=.65,
                  linked_object_mass=.65, linked_question_mass=.30,
                  linked_query_key_mass=.65)
    texts = {'fineweb': natural}
    for name, cls in (('existing_v10', SurfaceMixtureV10LinkedEvents),
                      ('bracket_bundle', SurfaceMixtureV10Bundled)):
        gen = cls(seed=args.seed, **kwargs)
        texts[name] = [render([gen.id2token[i] for i in
                               gen.sample_sequence(args.length)[0]])
                       for _ in range(args.n)]
    eligible = [i for i in range(args.n)
                if all(len(first_words(rows[i], 40)) >= 40
                       for rows in texts.values())]
    if not eligible:
        raise RuntimeError('No common 40-word form panel')
    result = {'config': {**vars(args), 'output': str(args.output),
                         'kwargs': kwargs},
              'n_common_40_word': len(eligible),
              'series': {name: {'forms': form_cues(rows),
                                'all_window_repetition': score(rows, 40),
                                'matched_40_word_repetition': score(
                                    [rows[i] for i in eligible], 40),
                                'matched_40_word_lexical': lexical(
                                    [rows[i] for i in eligible], 40)}
                         for name, rows in texts.items()},
              'scope': 'Fresh unselected FineWeb offsets, source prose not '
                       'saved. Generated and FineWeb streams use different '
                       'tokenizers, so 256 tokens are not equal-length texts; '
                       'matched 40-word metrics reduce but do not remove '
                       'this confound. Form and Zipf screens do not prove '
                       'semantic realism.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    for name, row in result['series'].items():
        print(name, row['forms'], row['matched_40_word_lexical'],
              flush=True)


if __name__ == '__main__':
    main()

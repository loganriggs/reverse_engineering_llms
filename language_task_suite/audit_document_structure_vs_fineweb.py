"""Independent FineWeb versus V8–V10 structural form screen.

Reports aggregates only: sentence openings, lengths, shared n-grams, and
layout. It does not save natural or generated document bodies, and it does
not infer semantic dependencies from punctuation alone.
"""

import argparse
import gc
import json
import os
import random
import re
import sys
from collections import Counter
from itertools import islice
from pathlib import Path

from datasets import load_dataset

from audit_surface_repetition import WORD, form_cues, score
from compare_v12_fineweb_fresh import first_words
from document_world_v8_discourse import draw_varied as draw_v8, realize as realize_v8
from document_world_v9_typed_lexicon import draw_lexicon, lexicalize_document
from document_world_v10_fact_acts import plan_acts, realize as realize_v10


SPLIT = re.compile(r'(?<=[.!?])\s+|\n+')
STRUCTURE_WORDS = ('the', 'of', 'to', 'and', 'was', 'in', 'for',
                   'report', 'owner', 'color', 'place', 'label')


def structure(texts):
    all_words = [[w.lower() for w in WORD.findall(text)] for text in texts]
    pooled = Counter(w for words in all_words for w in words)
    starts, sentence_lengths = Counter(), []
    first_starts = Counter()
    for text in texts:
        segments = [WORD.findall(span) for span in SPLIT.split(text)]
        segments = [words for words in segments if words]
        for j, words in enumerate(segments):
            start = words[0].lower()
            starts[start] += 1
            sentence_lengths.append(len(words))
            if j == 0:
                first_starts[start] += 1
    ngrams = Counter()
    document_frequency = Counter()
    for words in all_words:
        grams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
        ngrams.update(grams)
        document_frequency.update(set(grams))
    total_grams = sum(ngrams.values())
    mean_length = sum(sentence_lengths)/len(sentence_lengths)
    sorted_lengths = sorted(sentence_lengths)
    return dict(n=len(texts), word_types=len(pooled),
                top10_word_share=sum(c for _, c in pooled.most_common(10))
                /sum(pooled.values()),
                top10_words=[(w, round(c/sum(pooled.values()), 4))
                             for w, c in pooled.most_common(10)],
                structure_word_share={w: pooled[w]/sum(pooled.values())
                                      for w in STRUCTURE_WORDS},
                first40_repetition=score(texts, 40),
                form_cues=form_cues(texts),
                sentence_segments_per_doc=len(sentence_lengths)/len(texts),
                mean_segment_words=mean_length,
                median_segment_words=sorted_lengths[len(sorted_lengths)//2],
                sentence_start_types=len(starts),
                top5_sentence_start_share=sum(c for _, c in starts.most_common(5))
                /sum(starts.values()),
                top_sentence_starts=starts.most_common(12),
                first_sentence_start_types=len(first_starts),
                top5_trigram_share=sum(c for _, c in ngrams.most_common(5))
                /total_grams,
                repeated_across_docs_trigram_occurrence_share=sum(
                    count for gram, count in ngrams.items()
                    if document_frequency[gram] > 1)/total_grams)


def generated(n, seed, words):
    rng = random.Random(seed)
    texts = {key: [] for key in ('v8', 'lexical', 'acts', 'both')}
    draws = 0
    while len(texts['v8']) < n:
        draws += 1
        base = draw_v8(rng, syntax_entropy=1., source_delay_rate=0.,
                       extra_source_recap_rate=0.)
        world = base.latent
        world['fact_act_plan'] = plan_acts(world, .8)
        world['surface_lexicon'] = draw_lexicon(
            random.Random(world['realization_seed'] ^ 0xC019), 0.)
        v8 = realize_v8(world, syntax_entropy=1.)
        acts = realize_v10(world, syntax_entropy=1.)
        docs = dict(v8=v8, lexical=lexicalize_document(v8, world),
                    acts=acts, both=lexicalize_document(acts, world))
        crops = {key: first_words(doc.text(), words)
                 for key, doc in docs.items()}
        if any(len(WORD.findall(text)) < words for text in crops.values()):
            continue
        for key, text in crops.items():
            texts[key].append(text)
        assert draws <= 100*n
    return {key: structure(rows) for key, rows in texts.items()}, draws


def natural(skip, n, words):
    dataset = load_dataset('HuggingFaceFW/fineweb', name='sample-10BT',
                           split='train', streaming=True)
    stream = iter(dataset)
    try:
        rows = list(islice(stream, skip, skip+n*4))
    finally:
        if hasattr(stream, 'close'):
            stream.close()
    del dataset, stream
    gc.collect()
    crops = []
    for row in rows:
        text = first_words(row['text'], words)
        if len(WORD.findall(text)) >= words:
            crops.append(text)
        if len(crops) == n:
            break
    assert len(crops) == n
    return structure(crops), len(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--skip', type=int, default=40000)
    p.add_argument('--n', type=int, default=64)
    p.add_argument('--words', type=int, default=100)
    p.add_argument('--seed', type=int, default=94121)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    toy, generated_draws = generated(a.n, a.seed, a.words)
    reference, scanned = natural(a.skip, a.n, a.words)
    result = dict(config=dict(skip=a.skip, n=a.n, words=a.words,
                              generated_seed=a.seed,
                              natural_source='HuggingFaceFW/fineweb '
                                             'sample-10BT train'),
                  generated_draws=generated_draws,
                  natural_rows_scanned=scanned,
                  panels=dict(fineweb=reference, **toy),
                  caveat='Descriptive first-100-word form diagnostics. '
                         'Sentence segmentation is a simple regex; '
                         'headings, abbreviations, and web markup can '
                         'distort it. Generated arms share worlds and '
                         'eligibility; FineWeb is an independent panel. '
                         'No source text or document bodies saved.')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    for key, row in result['panels'].items():
        print(key, 'types', row['word_types'],
              'top10', round(row['top10_word_share'], 3),
              'repeat', round(row['first40_repetition'][
                  'repeated_word_fraction'], 3),
              'start_types', row['sentence_start_types'],
              'crossdoc_trigram', round(row[
                  'repeated_across_docs_trigram_occurrence_share'], 3),
              flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()

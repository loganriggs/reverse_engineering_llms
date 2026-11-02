"""Crude form comparison between world documents and frozen FineWeb windows.

This is a diagnostic, not a linguistic validity score: the windows are not
documents and sentence splitting is intentionally simple.
"""

import json
import re
import statistics as st
from collections import Counter
from pathlib import Path

import torch
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_ROWS, DEFAULT_TOKENIZER
from document_world import GENRES, sample_world


def words(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)


def summary(texts):
    all_words = [w.lower() for text in texts for w in words(text)]
    sentence_lengths = []
    for text in texts:
        for sentence in re.split(r'[.!?]+', text):
            n = len(words(sentence))
            if 5 <= n <= 80:
                sentence_lengths.append(n)
    counts = Counter(all_words)
    return dict(n_texts=len(texts), mean_words_per_text=round(len(all_words)/len(texts), 1),
                unique_words=len(counts), type_token_ratio=round(len(counts)/len(all_words), 4),
                sentence_count=len(sentence_lengths),
                mean_sentence_words=round(st.mean(sentence_lengths), 1),
                median_sentence_words=st.median(sentence_lengths),
                p90_sentence_words=sorted(sentence_lengths)[int(.9*len(sentence_lengths))],
                mean_commas_per_100_words=round(100*sum(t.count(',') for t in texts)/len(all_words), 2),
                mean_colons_per_100_words=round(100*sum(t.count(':') for t in texts)/len(all_words), 2))


def main():
    rows = torch.load(DEFAULT_ROWS, map_location='cpu', weights_only=True)
    tok = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    fineweb = [tok.decode(row.tolist(), skip_special_tokens=False) for row in rows]
    result = dict(fineweb_513_token_windows=summary(fineweb),
                  synthetic={GENRES[g]: summary([ex.text() for ex in
                    sample_world(1000, 703+g, genres=(g,))]) for g in range(3)},
                  caveat='FineWeb units are fixed token windows, not full documents; '
                         'naive regex sentence segmentation and word counts are only rough form diagnostics. '
                         'Synthetic examples end at a task answer and intentionally have small vocabularies.')
    path = Path(__file__).with_name('world_surface_comparison.json')
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

"""Compare actual rule-conditioned training windows to fresh FineWeb windows.

Only aggregate form statistics are written. Tensor caches and source text stay
local. This audits surfaces, not the semantic validity of the planted rules.
"""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import islice
import json
import os
from pathlib import Path
import random
import re
import sys

from datasets import load_dataset
import torch
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER


WORDS = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def cache_texts(row, tokenizer, n, seed, draws_per_source=1):
    rng = random.Random(seed)
    indices = list(range(0, len(row['x']), draws_per_source))
    rng.shuffle(indices)
    texts = []
    for index in indices[:n]:
        valid = int((row['y'][index] != -100).sum())
        texts.append(tokenizer.decode(row['x'][index, :valid].tolist()))
    assert len(texts) == n
    return texts


def stats(texts):
    words = [WORDS.findall(text.lower())[:100] for text in texts]
    words = [row for row in words if len(row) >= 40]
    flat = [word for row in words for word in row]
    counts = Counter(flat)
    trigrams = [list(zip(row, row[1:], row[2:])) for row in words]
    documents_with = Counter(gram for row in trigrams for gram in set(row))
    total_trigrams = sum(map(len, trigrams))
    first40 = [row[:40] for row in words]
    return {
        'windows_at_least_40_words': len(words),
        'alphabetic_word_positions': len(flat),
        'word_types': len(counts),
        'singleton_types': sum(value == 1 for value in counts.values()),
        'top10_word_share': sum(value for _, value in counts.most_common(10)) /
                             len(flat),
        'cross_document_repeated_trigram_occurrence_share':
            sum(documents_with[gram] > 1 for row in trigrams for gram in row) /
            total_trigrams,
        'first40_repeated_word_fraction': sum(
            len(row) - len(set(row)) for row in first40) /
            sum(map(len, first40)),
        'window_fraction_with_newline': sum('\n' in text for text in texts) /
                                        len(texts),
        'window_fraction_with_colon': sum(':' in text for text in texts) /
                                      len(texts),
        'window_fraction_with_parenthesis': sum(
            '(' in text or ')' in text for text in texts) / len(texts),
        'window_fraction_with_numbered_line': sum(bool(re.search(
            r'(?m)^\s{0,3}[1-9]\.[ \t]+\S', text)) for text in texts) /
            len(texts),
    }


def fineweb_texts(tokenizer, start, n, max_chars, n_ctx, seed):
    source = load_dataset('HuggingFaceFW/fineweb', name='sample-10BT',
                          split='train', streaming=True)
    stream = iter(source)
    rng = random.Random(seed)
    texts = []
    scanned = 0
    try:
        for datum in islice(stream, start, start + 3 * n):
            scanned += 1
            ids = tokenizer.encode(datum['text'][:max_chars],
                                   add_special_tokens=False).ids
            if len(ids) < 128:
                continue
            crop = rng.randrange(max(1, len(ids) - n_ctx + 1))
            texts.append(tokenizer.decode(ids[crop:crop + n_ctx]))
            if len(texts) == n:
                break
    finally:
        if hasattr(stream, 'close'):
            stream.close()
    if len(texts) != n:
        raise RuntimeError(f'Only {len(texts)}/{n} reference windows')
    return texts, scanned


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--nonce-cache', type=Path, required=True)
    p.add_argument('--list-cache', type=Path, required=True)
    p.add_argument('--bracket-cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--fineweb-start', type=int, default=350000)
    p.add_argument('--n', type=int, default=400)
    p.add_argument('--nonce-draws-per-source', type=int, default=8)
    p.add_argument('--max-chars', type=int, default=8000)
    p.add_argument('--n-ctx', type=int, default=256)
    p.add_argument('--seed', type=int, default=72401)
    a = p.parse_args()
    assert a.n % 4 == 0
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    nonce = torch.load(a.nonce_cache, map_location='cpu', weights_only=True)
    lists = torch.load(a.list_cache, map_location='cpu', weights_only=True)
    brackets = torch.load(a.bracket_cache, map_location='cpu', weights_only=True)
    by_kind = {
        'nonce': cache_texts(nonce['paired']['train']['left'], tokenizer,
                             a.n // 2, a.seed + 1,
                             a.nonce_draws_per_source),
        'list': cache_texts(lists['list']['train'][0], tokenizer,
                            a.n // 4, a.seed + 2),
        'bracket': cache_texts(brackets['train'], tokenizer,
                               a.n // 4, a.seed + 3),
    }
    reference, scanned = fineweb_texts(
        tokenizer, a.fineweb_start, a.n, a.max_chars, a.n_ctx, a.seed + 4)
    result = {'config': {key: str(value) if isinstance(value, Path) else value
                         for key, value in vars(a).items()},
              'reference_documents_scanned': scanned,
              'reference_offset_range': [a.fineweb_start,
                                         a.fineweb_start + scanned],
              'stats': {'training_mixture': stats(sum(by_kind.values(), [])),
                        'reference': stats(reference),
                        **{kind: stats(texts) for kind, texts in by_kind.items()}},
              'scope': 'First 100 alphabetic words of decoded 256-BPE-token '
                       'training windows, versus random-position crops from '
                       'fresh FineWeb source documents of >=128 BPE tokens. '
                       'Training mixture proportions are 50/25/25 by task '
                       'window, not natural task prevalence. The comparison '
                       'does not evaluate rule semantics or general web '
                       'dependency frequencies. No source prose saved.'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print({key: value for key, value in result['stats'].items()}, flush=True)
    # HF's streaming iterator can abort in native cleanup after the complete
    # aggregate has been flushed. Follow the existing cache-build convention.
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()

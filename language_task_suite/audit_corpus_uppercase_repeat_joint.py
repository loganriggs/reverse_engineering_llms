"""Screen uppercase surface recurrence plus a delimiter in FineWeb.

This is an opportunity screen for literal copying, not an entity oracle.
No source text or identifier strings are saved.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import islice
import json
import os
from pathlib import Path
import re
import sys

from datasets import load_dataset
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from corpus_transformed_delimiters import candidates as delimiter_candidates


IDENTIFIER = re.compile(r'(?<![A-Za-z])[A-Z]{3,5}(?![A-Za-z])')
WORD = re.compile(r'\b[A-Za-z]+\b')


def repeats(text, min_gap=20, max_gap=300):
    groups = defaultdict(list)
    for match in IDENTIFIER.finditer(text):
        groups[match.group()].append(match.span())
    result = []
    for name, spans in groups.items():
        for source, target in zip(spans, spans[1:]):
            gap = len(WORD.findall(text[source[1]:target[0]]))
            if min_gap <= gap <= max_gap:
                result.append((name, source, target, gap))
    return tuple(result)


def _token_index(offsets, position):
    return next((i for i, (start, end) in enumerate(offsets)
                 if start <= position < end), None)


def audit(skip, n, max_chars, max_joint_span, n_ctx):
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    source = load_dataset('HuggingFaceFW/fineweb',
                          name='sample-10BT', split='train', streaming=True)
    counts = Counter()
    stream = iter(source)
    try:
        for row in islice(stream, skip, skip+n):
            text = row['text'][:max_chars]
            counts['documents'] += 1
            names = repeats(text)
            brackets = delimiter_candidates(text)
            counts['uppercase_repeat'] += bool(names)
            counts['delimiter'] += bool(brackets)
            counts['both_anywhere'] += bool(names and brackets)
            if not names or not brackets:
                continue
            choices = [(name, source_span, target_span, opener, closer)
                       for name, source_span, target_span, _ in names
                       for opener, closer, _ in brackets
                       if max(target_span[1], closer)-
                          min(source_span[0], opener) <= max_joint_span]
            if not choices:
                continue
            counts['joint_within_char_span'] += 1
            enc = tokenizer.encode(text, add_special_tokens=False)
            span_aligned = False
            one_token_aligned = False
            for name, source_span, target_span, opener, closer in choices:
                indices = [_token_index(enc.offsets, pos)
                           for pos in (source_span[0], target_span[0],
                                       target_span[1]-1, opener, closer)]
                if any(i is None for i in indices):
                    continue
                source_i, target_first_i, target_last_i, open_i, close_i = indices
                if not (enc.offsets[target_last_i][1] == target_span[1] and
                        enc.offsets[close_i][1] == closer+1):
                    continue
                left = max(0, min(source_i, open_i)-32)
                if max(target_last_i, close_i)-left >= n_ctx:
                    continue
                span_aligned = True
                counts['span_target_token_length_'+str(
                    target_last_i-target_first_i+1)] += 1
                if target_first_i == target_last_i:
                    counts['joint_one_token_target'] += 1
                    one_token_aligned = True
                else:
                    counts['joint_multi_token_target'] += 1
            counts['joint_context_aligned_span'] += span_aligned
            counts['joint_one_token_aligned_documents'] += one_token_aligned
    finally:
        if hasattr(stream, 'close'):
            stream.close()
    if counts['documents'] != n:
        raise RuntimeError('Incomplete corpus panel')
    return {'config': {'skip': skip, 'n': n, 'max_chars': max_chars,
                       'max_joint_span': max_joint_span, 'n_ctx': n_ctx},
            'counts': dict(counts),
            'caveat': 'Uppercase recurrence is a literal surface pattern; '
                      'it does not certify one referent or a natural '
                      'counterfactual replacement. No model trained.'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--skip', type=int, default=84500)
    p.add_argument('--n', type=int, default=1000)
    p.add_argument('--max-chars', type=int, default=8000)
    p.add_argument('--max-joint-span', type=int, default=1200)
    p.add_argument('--n-ctx', type=int, default=320)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = audit(args.skip, args.n, args.max_chars,
                   args.max_joint_span, args.n_ctx)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(result['counts'], flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()

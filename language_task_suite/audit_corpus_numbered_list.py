"""Screen natural numbered-list continuation and bracket composition."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from itertools import islice
import json
import os
from pathlib import Path
import sys

from datasets import load_dataset
from tokenizers import Tokenizer

from audit_document_structure_vs_fineweb import structure
from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from audit_surface_repetition import WORD
from compare_v12_fineweb_fresh import first_words
from corpus_numbered_list_dgp import plans, realize
from corpus_numbered_list_validation import aligned, check, token_local_factorial


def site_window(doc, width=41):
    words = list(WORD.finditer(doc.text))
    if len(words) < width:
        return None
    target = doc.sites[-1].start
    rank = sum(match.start() < target for match in words)
    begin = max(0, min(rank-width//2, len(words)-width))
    return doc.text[words[begin].start():words[begin+width-1].end()]


def audit(skip, n, max_chars, n_ctx):
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    dataset = load_dataset('HuggingFaceFW/fineweb', name='sample-10BT',
                           split='train', streaming=True)
    counts = Counter()
    hashes = []
    source_first, edited_first = [], []
    source_window, edited_window = [], []
    stream = iter(dataset)
    try:
        for row in islice(stream, skip, skip+n):
            text = row['text'][:max_chars]
            counts['documents'] += 1
            list_plans = list(plans(text, max_chars=max_chars))
            if not list_plans:
                continue
            counts['list_triple'] += 1
            plan = list_plans[0]
            simple = {(shift, 'round'): realize(plan, shift)
                      for shift in (0, 1)}
            check(simple)
            counts['list_local_edit_valid'] += 1
            if all(aligned(doc, tokenizer, n_ctx)
                   for doc in simple.values()):
                counts['list_two_arm_bpe_context'] += 1
            source_first.append(first_words(simple[(0, 'round')].text, 100))
            edited_first.append(first_words(simple[(1, 'round')].text, 100))
            for doc, output in ((simple[(0, 'round')], source_window),
                                (simple[(1, 'round')], edited_window)):
                window = site_window(doc)
                if window:
                    output.append(window)
            joint = list(plans(text, with_bracket=True,
                               max_chars=max_chars))
            if not joint:
                continue
            counts['list_and_bracket_nearby'] += 1
            for plan in joint:
                arms = {(shift, style): realize(plan, shift, style)
                        for shift in (0, 1)
                        for style in ('round', 'square')}
                try:
                    check(arms)
                except ValueError:
                    continue
                if all(aligned(doc, tokenizer, n_ctx)
                       for doc in arms.values()):
                    counts['joint_four_arm_bpe_context'] += 1
                    counts['joint_strict_token_local'] += token_local_factorial(
                        arms, tokenizer)
                    hashes.append(hashlib.sha256(
                        row['text'].encode()).hexdigest())
                    break
    finally:
        if hasattr(stream, 'close'):
            stream.close()
    if counts['documents'] != n:
        raise RuntimeError('Incomplete FineWeb panel')
    return {'config': {'skip': skip, 'n': n, 'max_chars': max_chars,
                       'n_ctx': n_ctx},
            'counts': dict(counts),
            'joint_source_sha256': hashes,
            'first100_form': {
                'source': structure(source_first) if source_first else None,
                'shifted': structure(edited_first) if edited_first else None},
            'list_site41_form': {
                'source': structure(source_window) if source_window else None,
                'shifted': structure(edited_window) if edited_window else None},
            'caveat': 'List numbers and parentheses are local visible rules. '
                      'Other prose has no oracle. Source+edit form uses only '
                      'selected documents, not the full FineWeb distribution. '
                      'No natural source prose is exported.'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--skip', type=int, default=95000)
    p.add_argument('--n', type=int, default=1000)
    p.add_argument('--max-chars', type=int, default=8000)
    p.add_argument('--n-ctx', type=int, default=320)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = audit(a.skip, a.n, a.max_chars, a.n_ctx)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(result['counts'], flush=True)
    sys.stdout.flush()
    os._exit(0)  # datasets streaming can stall at interpreter exit


if __name__ == '__main__':
    main()

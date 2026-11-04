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
from sports_alias_bracket_dgp import visible_bracket_answer


def aligned(doc, tokenizer, n_ctx):
    encoding = tokenizer.encode(doc.text, add_special_tokens=False)
    indices = []
    first_marker = next((i for i, (start, end) in enumerate(encoding.offsets)
                         if start <= doc.plan.markers[0][0] < end), None)
    if first_marker is None:
        return False
    indices.append(first_marker)
    for site in doc.sites:
        found = [i for i, (start, end) in enumerate(encoding.offsets)
                 if start <= site.start < end and end == site.end]
        if len(found) != 1:
            return False
        indices.append(found[0])
        source = next((i for i, (start, end) in enumerate(encoding.offsets)
                       if start <= site.source_start < end), None)
        if source is None:
            return False
        indices.append(source)
    return max(indices)-max(0, min(indices)-32) < n_ctx


def token_local_factorial(arms, tokenizer):
    """Require edits to alter only their containing BPE tokens to target."""
    encodings = {arm: tokenizer.encode(doc.text, add_special_tokens=False)
                 for arm, doc in arms.items()}
    base = arms[(0, 'round')]
    baseline = encodings[(0, 'round')]
    target_char = base.sites[-1].start

    def index(enc, char):
        return next((i for i, (start, end) in enumerate(enc.offsets)
                     if start <= char < end), None)

    limit = index(baseline, target_char)
    if limit is None:
        return False
    for arm, doc in arms.items():
        enc = encodings[arm]
        if (index(enc, target_char) != limit or
                enc.offsets[:limit+1] != baseline.offsets[:limit+1]):
            return False
        changed_chars = [i for i, (a, b) in enumerate(
            zip(base.text, doc.text)) if a != b and i <= target_char]
        expected = {index(baseline, pos) for pos in changed_chars}
        if None in expected:
            return False
        actual = {i for i in range(limit+1)
                  if enc.ids[i] != baseline.ids[i]}
        if actual != expected:
            return False
    return True


def check(arms):
    base = arms[(0, 'round')]
    source = base.plan.source_text
    for (shift, style), doc in arms.items():
        assert len(doc.text) == len(source)
        changes = {i for i, (a, b) in enumerate(zip(source, doc.text))
                   if a != b}
        expected = ({start for start, _, _ in base.plan.markers}
                    if shift else set())
        if base.plan.bracket and style == 'square':
            expected.update(base.plan.bracket)
        assert changes == expected
        for site in doc.sites:
            assert doc.text[site.start:site.end] == site.answer
            if site.kind == 'bracket':
                assert visible_bracket_answer(doc.text[:site.start]) == \
                       site.answer
    assert arms[(0, 'round')].sites[-1].answer != \
           arms[(1, 'round')].sites[-1].answer
    if base.plan.bracket:
        assert arms[(0, 'round')].sites[0].answer != \
               arms[(0, 'square')].sites[0].answer


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

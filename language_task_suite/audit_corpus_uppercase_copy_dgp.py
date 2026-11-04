"""Text-free audit of sampled literal-code copying in FineWeb documents."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from itertools import islice
import json
import os
from pathlib import Path
import random
import sys

from datasets import load_dataset
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from corpus_uppercase_copy_dgp import WORD, candidates, codebook
from corpus_uppercase_copy_validation import aligned_pair, check_fixture


def crop(text, start_word, count):
    matches = list(WORD.finditer(text))
    if len(matches) < start_word + count:
        return None
    return text[matches[start_word].start():matches[start_word + count - 1].end()]


def around(text, position, words=41):
    matches = list(WORD.finditer(text))
    target = next((i for i, match in enumerate(matches)
                   if match.start() <= position < match.end()), None)
    if target is None or len(matches) < words:
        return None
    begin = max(0, min(target - words // 2, len(matches) - words))
    return crop(text, begin, words)


def form(texts):
    rows = [[match.group().lower() for match in WORD.finditer(text)]
            for text in texts]
    pooled = Counter(word for words in rows for word in words)
    grams = Counter()
    document_frequency = Counter()
    for words in rows:
        observed = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
        grams.update(observed)
        document_frequency.update(set(observed))
    total = sum(pooled.values())
    return {'n': len(texts), 'word_types': len(pooled),
            'top10_word_share': sum(value for _, value in pooled.most_common(10)) / total,
            'cross_doc_repeated_trigram_share': sum(
                count for gram, count in grams.items()
                if document_frequency[gram] > 1) / sum(grams.values())}


def audit(skip, n, max_chars, seed, n_ctx, print_snippets=False):
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    check_fixture(tokenizer)
    source = load_dataset('HuggingFaceFW/fineweb', name='sample-10BT',
                          split='train', streaming=True)
    stream = iter(source)
    rng = random.Random(seed)
    counts = Counter()
    hashes = []
    gaps = []
    first100 = {'source': [], 'left': [], 'right': []}
    target41 = {'source': [], 'left': [], 'right': []}
    try:
        for offset, row in enumerate(islice(stream, skip, skip + n), skip):
            counts['source_documents'] += 1
            text = row['text'][:max_chars]
            options = list(candidates(text))
            counts['candidate_documents'] += bool(options)
            counts['candidate_pairs'] += len(options)
            rng.shuffle(options)
            selected = None
            for plan in options:
                pair, reason = aligned_pair(plan, tokenizer, n_ctx, rng)
                if pair is not None:
                    selected = pair
                    break
            if selected is None:
                continue
            counts['aligned_documents'] += 1
            counts['changed_prefix_tokens'] += selected['changed_prefix_tokens']
            left, right = selected['left_example'], selected['right_example']
            plan = left.plan
            assert plan.source_text == text
            assert left.code != right.code
            assert left.text[:left.answer_span[0]] != \
                   right.text[:right.answer_span[0]]
            hashes.append({'offset': offset,
                           'sha256': hashlib.sha256(text.encode()).hexdigest()})
            gaps.append(plan.gap_words)
            for key, version in (('source', text), ('left', left.text),
                                 ('right', right.text)):
                prefix = crop(version, 0, 100)
                if prefix:
                    first100[key].append(prefix)
                site = around(version, left.answer_span[0])
                if site:
                    target41[key].append(site)
            if print_snippets and counts['aligned_documents'] <= 12:
                a, b = left.previous_span, left.answer_span
                print('offset', offset, 'source_code', plan.source_code,
                      'sampled', left.code, right.code, 'gap', plan.gap_words)
                print('source context', text[max(0, a[0] - 70):a[1] + 70]
                      .replace('\n', ' '))
                print('target context', text[max(0, b[0] - 70):b[1] + 70]
                      .replace('\n', ' '))
    finally:
        if hasattr(stream, 'close'):
            stream.close()
    assert counts['source_documents'] == n
    return {'source': 'HuggingFaceFW/fineweb sample-10BT train',
            'offsets': [skip, skip + n], 'max_chars': max_chars,
            'n_ctx': n_ctx, 'seed': seed,
            'codebook_size': len(codebook(tokenizer)),
            'counts': dict(counts), 'selected_source_hashes': hashes,
            'gap_words': {'min': min(gaps), 'max': max(gaps),
                          'mean': sum(gaps) / len(gaps)} if gaps else None,
            'first100_form': {key: form(value) for key, value in first100.items()
                              if value},
            'target41_form': {key: form(value) for key, value in target41.items()
                              if value},
            'caveat': 'This samples a literal surface code, not an entity '
                      'identity or whole-document oracle. Acronym '
                      'substitution can make source propositions false. '
                      'Form comparisons are on the selected subset; '
                      'human semantic/naturalness review is required '
                      'before model training. No source prose saved.'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--skip', type=int, default=84500)
    p.add_argument('--n', type=int, default=1000)
    p.add_argument('--max-chars', type=int, default=8000)
    p.add_argument('--n-ctx', type=int, default=256)
    p.add_argument('--seed', type=int, default=71901)
    p.add_argument('--print-snippets', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = audit(a.skip, a.n, a.max_chars, a.seed,
                   a.n_ctx, a.print_snippets)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(result['counts'], result['gap_words'], flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()

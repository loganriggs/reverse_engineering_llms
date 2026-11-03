"""Document-level yield for natural-parenthetical style resampling."""

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from itertools import islice
from pathlib import Path

from datasets import load_dataset

from corpus_transformed_delimiters import candidates


def audit(skip, n, max_chars):
    source = load_dataset('HuggingFaceFW/fineweb', name='sample-10BT',
                          split='train', streaming=True)
    stream = iter(source)
    counts, hashes = Counter(), set()
    try:
        for row in islice(stream, skip, skip+n):
            text = row['text'][:max_chars]
            pairs = candidates(text)
            counts['documents'] += 1
            counts['eligible_documents'] += bool(pairs)
            counts['eligible_pairs'] += len(pairs)
            hashes.add(hashlib.sha256(text.encode()).hexdigest())
    finally:
        if hasattr(stream, 'close'):
            stream.close()
    assert counts['documents'] == n
    return dict(config=dict(skip=skip, n=n, max_chars=max_chars,
                            source='HuggingFaceFW/fineweb sample-10BT train'),
                counts=dict(counts), unique_exact_text_hashes=len(hashes),
                caveat='Eligibility screen only. No source prose saved; '
                       'not evidence of model learning or semantic realism '
                       'of replacing round with square delimiters.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--skip', type=int, default=43000)
    p.add_argument('--n', type=int, default=500)
    p.add_argument('--max-chars', type=int, default=8000)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = audit(args.skip, args.n, args.max_chars)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(result['counts'], flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()

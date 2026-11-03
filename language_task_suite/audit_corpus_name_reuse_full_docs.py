"""Document-level yield screen for a corpus-transformed name-reuse rule.

Streams FineWeb, saves only aggregate counts and hashes, and exits explicitly
because the datasets streaming worker can outlive the final iterator.
"""

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from itertools import islice
from pathlib import Path

from datasets import load_dataset

from corpus_transformed_name_reuse import candidates


def audit(skip, n, max_chars):
    dataset = load_dataset('HuggingFaceFW/fineweb', name='sample-10BT',
                           split='train', streaming=True)
    stream = iter(dataset)
    counts = Counter()
    text_hashes = set()
    try:
        for row in islice(stream, skip, skip+n):
            text = row['text'][:max_chars]
            text_hashes.add(hashlib.sha256(text.encode()).hexdigest())
            counts['documents'] += 1
            for policy in ('strict_full_name', 'first_name'):
                options, _ = candidates(text, identity_policy=policy)
                counts[policy+'_eligible'] += bool(options)
                counts[policy+'_sites'] += len(options)
    finally:
        if hasattr(stream, 'close'):
            stream.close()
    assert counts['documents'] == n
    return dict(config=dict(skip=skip, n=n, max_chars=max_chars,
                            source='HuggingFaceFW/fineweb sample-10BT train'),
                counts=dict(counts), unique_exact_text_hashes=len(text_hashes),
                caveat='Eligibility only; no neural model, semantic manual '
                       'review, or source prose saved. The loose policy can '
                       'conflate people with the same first name.')


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

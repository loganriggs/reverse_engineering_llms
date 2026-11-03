"""Yield screen for topic-matched natural prose as report-DGP carriers.

Only counts and hashes are saved. Eligibility is lexical, not proof that a
passage can support inserted synthetic facts without contradiction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from itertools import islice
from pathlib import Path

from datasets import load_dataset

from project_report_dgp import DOMAINS, WORD


PROJECTS = tuple(row[0] for row in DOMAINS)
PROJECT_PATTERN = {word: re.compile(r'\b'+re.escape(word)+r's?\b', re.I)
                   for word in PROJECTS}
REPAIR = re.compile(r'\b(?:repair(?:s|ed|ing)?|maintenance|inspection|'
                    r'closed|closure|reopen(?:ed|ing)?|renovation|'
                    r'damage(?:d)?|broken|faulty)\b', re.I)


def audit(skip, documents, max_chars):
    source = load_dataset('HuggingFaceFW/fineweb', name='sample-10BT',
                          split='train', streaming=True)
    stream = iter(source)
    counts = Counter()
    hashes = set()
    try:
        for row in islice(stream, skip, skip+documents):
            counts['source_documents'] += 1
            text = row['text'][:max_chars]
            if len(WORD.findall(text)) < 200:
                counts['too_short'] += 1
                continue
            counts['long_documents'] += 1
            matches = {word: len(pattern.findall(text))
                       for word, pattern in PROJECT_PATTERN.items()}
            present = [word for word, n in matches.items() if n]
            trigger_count = len(REPAIR.findall(text))
            if present:
                counts['project_mention'] += 1
            if trigger_count >= 2:
                counts['two_repair_terms'] += 1
            if not present or trigger_count < 2:
                continue
            counts['project_and_repair'] += 1
            for word in present:
                counts[f'project/{word}'] += 1
            if max(matches.values()) >= 2:
                counts['repeated_project_word'] += 1
            if text.count('\n') >= 2:
                counts['two_newlines'] += 1
            if max(matches.values()) >= 2 and text.count('\n') >= 2:
                counts['repeated_project_and_layout'] += 1
            hashes.add(hashlib.sha256(text.encode()).hexdigest())
    finally:
        if hasattr(stream, 'close'):
            stream.close()
    assert counts['source_documents'] == documents
    return dict(config=dict(skip=skip, documents=documents,
                            max_chars=max_chars,
                            source='HuggingFaceFW/fineweb sample-10BT train'),
                counts=dict(counts),
                eligible_exact_hashes=len(hashes),
                caveat='Lexical carrier screen only. A matched project '
                       'word can have another meaning; no entity or '
                       'contradiction validation is implied.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--skip', type=int, default=100000)
    p.add_argument('--documents', type=int, default=10000)
    p.add_argument('--max-chars', type=int, default=8000)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = audit(args.skip, args.documents, args.max_chars)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(result['counts'], flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()

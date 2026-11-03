"""Compare discourse-construction surfaces in FineWeb and proposition V2.

These are deliberately lexical screens, not semantic parses. They inform
renderer design; no regex match is treated as a verified reasoning site.
Only aggregate counts are saved. Source text is never written to Git.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from itertools import islice
from pathlib import Path

from datasets import load_dataset

from document_proposition_plan import sample_plan
from document_proposition_surface import realize


WORD = re.compile(r"\b[A-Za-z]+(?:'[A-Za-z]+)?\b")
SENTENCE = re.compile(r'[^.!?\n]+[.!?]?')
CONSTRUCTIONS = {
    'attribution': {
        'according_to': r'\baccording to\b',
        'reported_by': r'\breported by\b',
        'said': r'\bsaid\b',
        'reported': r'\breported\b',
        'announced': r'\bannounced\b',
        'confirmed': r'\bconfirmed\b',
        'told': r'\btold\b',
        'wrote': r'\bwrote\b',
        'noted': r'\bnoted\b',
        'claimed': r'\bclaimed\b',
    },
    'change': {
        'changed_from': r'\bchanged from\b',
        'went_from': r'\bwent from\b',
        'rose_from': r'\brose from\b',
        'fell_from': r'\bfell from\b',
        'increased_from': r'\bincreased from\b',
        'decreased_from': r'\bdecreased from\b',
        'previously': r'\bpreviously\b',
        'used_to_be': r'\bused to be\b',
        'now': r'\bnow\b',
    },
    'comparison': {
        'compared_with': r'\bcompared with\b',
        'compared_to': r'\bcompared to\b',
        'in_contrast': r'\bin contrast\b',
        'whereas': r'\bwhereas\b',
        'while': r'\bwhile\b',
        'same_as': r'\bsame as\b',
        'different_from': r'\bdifferent from\b',
        'both': r'\bboth\b',
    },
    'quantity': {
        'a_total_of': r'\ba total of\b',
        'there_were': r'\bthere were\b',
        'number_of': r'\bnumber of\b',
        'count_of': r'\bcount of\b',
        'more_than': r'\bmore than\b',
        'less_than': r'\bless than\b',
    },
}
PATTERNS = {family: {name: re.compile(pattern, re.I)
                     for name, pattern in rows.items()}
            for family, rows in CONSTRUCTIONS.items()}


def _prefix(text, max_words):
    words = list(WORD.finditer(text))
    if len(words) <= max_words:
        return text
    return text[:words[max_words-1].end()]


def _record(text, counts):
    counts['documents'] += 1
    counts['words'] += len(WORD.findall(text))
    family_docs = set()
    construction_docs = set()
    for sentence in SENTENCE.findall(text):
        nwords = len(WORD.findall(sentence))
        if nwords < 3:
            continue
        if nwords > 60:
            counts['sentences_over_60_words'] += 1
            continue
        counts['screened_sentences'] += 1
        counts['sentences_under_8_words' if nwords < 8 else
               'sentences_8_to_60_words'] += 1
        matched_families = set()
        for family, rows in PATTERNS.items():
            for name, pattern in rows.items():
                if pattern.search(sentence):
                    counts[f'construction/{family}/{name}'] += 1
                    construction_docs.add((family, name))
                    matched_families.add(family)
        for family in matched_families:
            counts[f'family_sentences/{family}'] += 1
            family_docs.add(family)
        if len(matched_families) >= 2:
            counts['sentences_with_multiple_families'] += 1
        if matched_families:
            counts['sentences_with_any_family'] += 1
    for family in family_docs:
        counts[f'family_documents/{family}'] += 1
    for family, name in construction_docs:
        counts[f'construction_documents/{family}/{name}'] += 1
    counts[f'families_per_doc/{len(family_docs)}'] += 1


def _summary(counts):
    words = counts['words']; sentences = counts['screened_sentences']
    by_family = {}
    for family, rows in CONSTRUCTIONS.items():
        by_family[family] = dict(
            documents=counts[f'family_documents/{family}'],
            sentence_rate=(counts[f'family_sentences/{family}']/sentences
                           if sentences else 0),
            per_1000_words=1000*counts[f'family_sentences/{family}']/words,
            constructions={name: dict(
                sentences=counts[f'construction/{family}/{name}'],
                documents=counts[
                    f'construction_documents/{family}/{name}'])
                for name in rows})
    return dict(documents=counts['documents'], words=words,
                screened_sentences=sentences,
                sentences_under_8_words=counts['sentences_under_8_words'],
                sentences_8_to_60_words=counts['sentences_8_to_60_words'],
                sentences_over_60_words=counts['sentences_over_60_words'],
                sentences_with_any_family=counts['sentences_with_any_family'],
                sentences_with_multiple_families=counts[
                    'sentences_with_multiple_families'],
                families_per_doc={str(i): counts[f'families_per_doc/{i}']
                                  for i in range(5)},
                by_family=by_family)


def audit(skip, documents, max_words, synthetic_seed, surface_version):
    natural = Counter()
    source = load_dataset('HuggingFaceFW/fineweb', name='sample-10BT',
                          split='train', streaming=True)
    stream = iter(source)
    try:
        for row in islice(stream, skip, skip+documents):
            _record(_prefix(row['text'], max_words), natural)
    finally:
        if hasattr(stream, 'close'):
            stream.close()
    assert natural['documents'] == documents

    synthetic = Counter()
    semantic = Counter()
    rng = random.Random(synthetic_seed)
    for index in range(documents):
        plan = sample_plan(rng, planning_style='threaded', n_moves=11)
        kinds = {move['kind'] for move in plan['moves']}
        for kind in kinds:
            semantic[f'documents_with/{kind}'] += 1
        semantic[f'move_kinds_per_doc/{len(kinds)}'] += 1
        semantic[f'forms/{plan["form"]}'] += 1
        document = realize(plan, seed=synthetic_seed+index,
                           lexical_exponent=1.05,
                           surface_version=surface_version)
        _record(_prefix(document.text(), max_words), synthetic)
    return dict(config=dict(source='FineWeb sample-10BT train', skip=skip,
                            documents=documents, max_words=max_words,
                            synthetic=f'threaded proposition V{surface_version}',
                            synthetic_seed=synthetic_seed),
                natural=_summary(natural),
                synthetic=_summary(synthetic),
                synthetic_semantic_plan=dict(semantic),
                caveat='Lexical patterns are not semantic parses. '
                       'FineWeb pages and V2 samples have different '
                       'content, genre mix, and length distributions. '
                       'The comparison diagnoses surface diversity only.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--skip', type=int, default=90000)
    p.add_argument('--documents', type=int, default=2000)
    p.add_argument('--max-words', type=int, default=500)
    p.add_argument('--synthetic-seed', type=int, default=91827)
    p.add_argument('--surface-version', type=int, choices=(2, 3), default=2)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = audit(args.skip, args.documents, args.max_words,
                   args.synthetic_seed, args.surface_version)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    for key in ('natural', 'synthetic'):
        value = result[key]
        print(key, value['documents'], value['words'],
              value['families_per_doc'], flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()

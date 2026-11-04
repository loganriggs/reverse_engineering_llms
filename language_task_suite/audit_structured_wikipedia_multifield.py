"""Screen clean Wikipedia leads for multiple visible infobox-grounded sites.

Literal overlap is a necessary, not sufficient, condition for controlled
editing. This script writes aggregate counts only; it does not export prose.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re

import pyarrow.parquet as pq
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER


WORDS = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
UNSAFE_LABELS = {'coordinates', 'image', 'website', 'url', 'logo',
                 'signature', 'caption', 'image size'}


def walk_fields(node):
    if isinstance(node, list):
        for part in node:
            yield from walk_fields(part)
    elif isinstance(node, dict):
        if node.get('type') == 'field':
            label = node.get('name')
            value = node.get('value')
            if isinstance(label, str) and isinstance(value, str):
                yield label.strip(), value.strip()
        for part in node.values():
            if isinstance(part, (list, dict)):
                yield from walk_fields(part)


def clean_field(label, value):
    label = re.sub(r'\s+', ' ', label).strip()
    value = re.sub(r'\s+', ' ', value).strip()
    if (not label or not value or label.lower() in UNSAFE_LABELS or
            not 2 <= len(label) <= 40 or not 3 <= len(value) <= 36 or
            len(value.split()) > 5 or '\n' in label or '\n' in value or
            re.search(r'https?://|\[[^]]+\]|\([^)]{30,}\)', value)):
        return None
    return label, value


def literal_mentions(text, value):
    return list(re.finditer(r'(?<!\w)' + re.escape(value) + r'(?!\w)',
                           text, re.I))


def candidate(row, *, max_words=170):
    abstract = row.get('abstract') or ''
    raw_infobox = row.get('infoboxes')
    if not raw_infobox or len(WORDS.findall(abstract)) < 45:
        return []
    try:
        infoboxes = json.loads(raw_infobox)
    except (ValueError, TypeError):
        return []
    fields = []
    seen = set()
    for label, value in walk_fields(infoboxes):
        cleaned = clean_field(label, value)
        if cleaned is None:
            continue
        label, value = cleaned
        key = (label.casefold(), value.casefold())
        if key in seen:
            continue
        seen.add(key)
        matches = literal_mentions(abstract, value)
        if len(matches) != 1:
            continue
        lo, hi = matches[0].span()
        if len(WORDS.findall(abstract[:lo])) > max_words:
            continue
        fields.append({'label': label, 'value': value, 'span': (lo, hi)})
    result = []
    for field in fields:
        lo, hi = field['span']
        if any(other is not field and
               not (hi <= other['span'][0] or other['span'][1] <= lo)
               for other in fields):
            continue
        result.append(field)
    return result


def sampled_rows(parquet, scan_rows=20000, seed=74401):
    file = pq.ParquetFile(parquet)
    rng = random.Random(seed)
    groups = list(range(file.num_row_groups))
    rng.shuffle(groups)
    visited = 0
    for group in groups:
        table = file.read_row_group(group,
                                    columns=['name', 'abstract', 'infoboxes'])
        rows = table.to_pylist()
        rng.shuffle(rows)
        for row in rows[:min(256, len(rows))]:
            yield row
            visited += 1
            if visited >= scan_rows:
                return


def audit(parquet, *, scan_rows=20000, seed=74401):
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    counts = Counter()
    labels, label_pairs = Counter(), Counter()
    examples = []
    for row in sampled_rows(parquet, scan_rows, seed):
        counts['scanned'] += 1
        fields = candidate(row)
        if not fields:
            continue
        counts['one_or_more'] += 1
        counts['two_or_more'] += len(fields) >= 2
        counts['three_or_more'] += len(fields) >= 3
        counts['matched_fields'] += len(fields)
        if len(fields) < 2:
            continue
        for field in fields:
            labels[field['label'].casefold()] += 1
        label_pairs[tuple(sorted((field['label'].casefold()
                                  for field in fields[:2])))] += 1
        prefix = '\n'.join(field['label'] + ': ' + field['value']
                           for field in fields[:3]) + '\n\n'
        enc = tokenizer.encode(prefix + row['abstract'])
        offsets = enc.offsets
        site_ids = []
        for field in fields[:3]:
            lo = len(prefix) + field['span'][0]
            hi = len(prefix) + field['span'][1]
            overlapping = [i for i, (a, b) in enumerate(offsets)
                           if a < hi and b > lo]
            if not overlapping:
                raise AssertionError('Lost a selected mention')
            site_ids.append(overlapping[-1])
        counts['two_sites_under_256'] += all(i < 256
                                              for i in site_ids[:2])
        counts['three_sites_under_256'] += (
            len(site_ids) >= 3 and all(i < 256 for i in site_ids[:3]))
        counts['full_document_under_256'] += len(enc.ids) <= 256
        if len(examples) < 12:
            examples.append({'title': row['name'],
                             'field_labels': [x['label'] for x in fields],
                             'field_values': [x['value'] for x in fields],
                             'abstract': row['abstract'][:500]})
    return {'source': 'wikimedia/structured-wikipedia enwiki shard 0',
            'seed': seed, 'scan_rows': scan_rows,
            'counts': dict(counts),
            'top_field_labels': labels.most_common(30),
            'top_field_pairs': [[list(k), v]
                                for k, v in label_pairs.most_common(20)],
            'local_examples_for_manual_review': examples,
            'caveat': 'Exact value overlap does not verify field semantics. '
                      'No candidate is approved for counterfactual editing '
                      'or model training by this screen.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parquet', type=Path, required=True)
    parser.add_argument('--scan-rows', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=74401)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = audit(args.parquet, scan_rows=args.scan_rows, seed=args.seed)
    # Source prose is for local inspection only, never the committed aggregate.
    examples = out.pop('local_examples_for_manual_review')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + '\n')
    print(out['counts'])
    for example in examples[:5]:
        print(example['title'], example['field_labels'][:3],
              example['field_values'][:3], flush=True)


if __name__ == '__main__':
    main()

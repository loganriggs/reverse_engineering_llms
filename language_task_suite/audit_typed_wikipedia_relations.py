"""Count clause-witnessed infobox relations in leads and later paragraphs."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random

import pyarrow.parquet as pq

from audit_document_structure_vs_fineweb import structure
from audit_structured_wikipedia_multifield import (
    WORDS, clean_field, walk_fields,
)
from audit_structured_wikipedia_section_sites import paragraphs
from typed_wikipedia_relations import evidence


def first_words(text, n):
    matches = list(WORDS.finditer(text))
    return text[:matches[n-1].end()] if len(matches) >= n else None


def paragraph_evidence(text, fields, title, require_title):
    found = []
    for field in fields:
        witness = evidence(text, field, title=title,
                           require_title=require_title)
        if witness and all(
                witness['label'].casefold() != x['label'].casefold()
                for x in found):
            found.append(witness)
    return found


def audit(parquet, *, scan_rows=10000, seed=74601,
          form_words=40):
    file = pq.ParquetFile(parquet)
    rng = random.Random(seed)
    groups = list(range(file.num_row_groups))
    rng.shuffle(groups)
    counts = Counter()
    labels = {'lead': Counter(), 'later': Counter()}
    pair_labels = {'lead': Counter(), 'later': Counter()}
    forms = {'lead': [], 'later': []}
    examples = []
    for group in groups:
        rows = file.read_row_group(
            group, columns=['name', 'abstract', 'infoboxes', 'sections']
        ).to_pylist()
        rng.shuffle(rows)
        for row in rows[:min(256, len(rows))]:
            if counts['scanned'] >= scan_rows:
                break
            counts['scanned'] += 1
            if not row['infoboxes']:
                continue
            try:
                infoboxes = json.loads(row['infoboxes'])
            except ValueError:
                continue
            fields = []
            for label, value in walk_fields(infoboxes):
                cleaned = clean_field(label, value)
                if cleaned:
                    fields.append({'label': cleaned[0],
                                   'value': cleaned[1]})
            if not fields:
                continue
            texts = [('lead', row['abstract'] or '')]
            if row['sections']:
                try:
                    sections = json.loads(row['sections'])
                except ValueError:
                    sections = []
                texts.extend(('later', text) for text in paragraphs(sections)
                             if 40 <= len(WORDS.findall(text)) <= 220)
            article_multi = {'lead': False, 'later': False}
            for source, text in texts:
                if len(WORDS.findall(text)) < 40:
                    continue
                found = paragraph_evidence(text, fields, row['name'],
                                           require_title=True)
                if not found:
                    continue
                counts[source + '_one_witness'] += 1
                for witness in found:
                    labels[source][witness['label'].casefold()] += 1
                if len(found) < 2:
                    continue
                counts[source + '_two_witness_paragraphs'] += 1
                article_multi[source] = True
                pair = tuple(sorted(x['label'].casefold()
                                    for x in found[:2]))
                pair_labels[source][pair] += 1
                if len(forms[source]) < 512:
                    crop = first_words(text, form_words)
                    if crop:
                        forms[source].append(crop)
                if len(examples) < 20:
                    examples.append((row['name'], source,
                                     tuple((x['label'], x['value'])
                                           for x in found[:3]),
                                     text[:450]))
            for source, yes in article_multi.items():
                counts[source + '_articles_with_two'] += yes
        if counts['scanned'] >= scan_rows:
            break
    result = {'source': 'wikimedia/structured-wikipedia enwiki shard 0',
              'seed': seed, 'scan_rows': scan_rows,
              'counts': dict(counts),
              'labels': {k: v.most_common(30) for k, v in labels.items()},
              'label_pairs': {k: [[list(pair), count]
                                  for pair, count in v.most_common(30)]
                              for k, v in pair_labels.items()},
              'form_n': {k: len(v) for k, v in forms.items()},
              'form': {k: structure(v) if v else None
                       for k, v in forms.items()},
              'caveat': 'Lexical clause witnesses plus an article-title '
                        'anchor are conservative candidates, not a full '
                        'semantic entailment or safe edit certificate. '
                        'No training data is approved by this audit.'}
    return result, examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parquet', type=Path, required=True)
    parser.add_argument('--scan-rows', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=74601)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result, examples = audit(args.parquet, scan_rows=args.scan_rows,
                             seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(result['counts'], flush=True)
    for example in examples[:8]:
        print(example, flush=True)


if __name__ == '__main__':
    main()

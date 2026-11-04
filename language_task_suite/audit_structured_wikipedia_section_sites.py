"""Check whether later article paragraphs carry multiple infobox facts."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random

import pyarrow.parquet as pq
from tokenizers import Tokenizer

from audit_document_structure_vs_fineweb import structure
from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from audit_structured_wikipedia_multifield import (
    WORDS, clean_field, literal_mentions, walk_fields,
)


def paragraphs(node, in_abstract=False):
    if isinstance(node, list):
        for child in node:
            yield from paragraphs(child, in_abstract)
    elif isinstance(node, dict):
        abstract = in_abstract or (
            node.get('type') == 'section' and
            node.get('name') == 'Abstract')
        if node.get('type') == 'paragraph' and not abstract:
            value = node.get('value')
            if isinstance(value, str):
                yield value
        for child in node.get('has_parts') or []:
            yield from paragraphs(child, abstract)


def first_words(text, n):
    matches = list(WORDS.finditer(text))
    return text[:matches[n-1].end()] if len(matches) >= n else None


def audit(parquet, *, scan_rows=10000, seed=74501,
          form_words=40, panel_size=512):
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    file = pq.ParquetFile(parquet)
    rng = random.Random(seed)
    groups = list(range(file.num_row_groups))
    rng.shuffle(groups)
    counts = Counter()
    selected, general = [], []
    for group in groups:
        rows = file.read_row_group(
            group, columns=['name', 'infoboxes', 'sections']).to_pylist()
        rng.shuffle(rows)
        for row in rows[:min(256, len(rows))]:
            if counts['scanned'] >= scan_rows:
                break
            counts['scanned'] += 1
            if not row['sections']:
                continue
            try:
                sections = json.loads(row['sections'])
            except ValueError:
                continue
            prose = [text for text in paragraphs(sections)
                     if 40 <= len(WORDS.findall(text)) <= 220]
            counts['articles_with_later_paragraph'] += bool(prose)
            if prose and len(general) < panel_size:
                general.append(first_words(rng.choice(prose), form_words))
            if not row['infoboxes'] or not prose:
                continue
            try:
                infobox = json.loads(row['infoboxes'])
            except ValueError:
                continue
            fields = []
            for label, value in walk_fields(infobox):
                cleaned = clean_field(label, value)
                if cleaned:
                    fields.append(cleaned)
            for text in prose:
                matches = [(label, value, literal_mentions(text, value))
                           for label, value in fields]
                matches = [(label, value, found[0].span())
                           for label, value, found in matches
                           if len(found) == 1]
                if len(matches) < 2:
                    continue
                distinct = []
                for label, value, span in matches:
                    if label.casefold() in [x[0].casefold()
                                             for x in distinct]:
                        continue
                    if any(not (span[1] <= x[2][0] or x[2][1] <= span[0])
                           for x in distinct):
                        continue
                    distinct.append((label, value, span))
                    if len(distinct) == 2:
                        break
                if len(distinct) < 2:
                    continue
                counts['two_field_paragraphs'] += 1
                prefix = (row['name'] + '\n' +
                          '\n'.join(label + ': ' + value
                                    for label, value, _ in distinct) + '\n\n')
                enc = tokenizer.encode(prefix + text)
                counts['two_field_paragraphs_under_256'] += len(enc.ids) <= 256
                if len(selected) < panel_size:
                    selected.append(first_words(text, form_words))
                break
        if counts['scanned'] >= scan_rows:
            break
    general = [text for text in general if text]
    selected = [text for text in selected if text]
    n = min(len(general), len(selected), panel_size)
    return {'source': 'wikimedia/structured-wikipedia enwiki shard 0',
            'seed': seed, 'form_words': form_words,
            'counts': dict(counts),
            'selected_form': structure(selected[:n]) if n else None,
            'general_later_paragraph_form':
                structure(general[:n]) if n else None,
            'form_n': n,
            'caveat': 'Literal matches only; no edits or semantic '
                      'faithfulness claims. Later paragraph may depend '
                      'on omitted article context.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parquet', type=Path, required=True)
    parser.add_argument('--scan-rows', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=74501)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = audit(args.parquet, scan_rows=args.scan_rows, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + '\n')
    print(out['counts'])


if __name__ == '__main__':
    main()

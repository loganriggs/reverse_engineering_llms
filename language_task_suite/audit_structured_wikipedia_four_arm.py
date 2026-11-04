"""Check a 2x2 multi-field empirical-language DGP candidate."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

from tokenizers import Tokenizer

from audit_document_structure_vs_fineweb import structure
from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from audit_structured_wikipedia_multifield import (
    candidate, sampled_rows,
)
from structured_wikipedia_multifield_dgp import make_four_arm


def first_words(text, n):
    import re
    matches = list(re.finditer(r"[A-Za-z]+(?:'[A-Za-z]+)?", text))
    return text[:matches[n-1].end()] if len(matches) >= n else None


def choose_alternate(rng, field, pool, abstract):
    choices = pool[field['label'].casefold()]
    same_width = [value for value in choices
                  if len(value.split()) == len(field['value'].split()) and
                  value.casefold() != field['value'].casefold()]
    rng.shuffle(same_width)
    for value in same_width[:40]:
        if value.casefold() not in abstract.casefold():
            return value
    return None


def audit(parquet, *, scan_rows=20000, seed=74451,
          panel_size=512, form_words=40, natural_preview=None):
    rows = list(sampled_rows(parquet, scan_rows, seed))
    rng = random.Random(seed)
    pool = defaultdict(set)
    fields_by_row = []
    for row in rows:
        fields = candidate(row)
        fields_by_row.append(fields)
        for field in fields:
            pool[field['label'].casefold()].add(field['value'])
    pool = {key: sorted(values) for key, values in pool.items()}
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    counts = Counter(scanned=len(rows))
    original, edited = [], []
    unselected = [first_words(row['abstract'] or '', form_words)
                  for row in rows]
    unselected = [text for text in unselected if text]
    pairs = Counter()
    examples = []
    for row, fields in zip(rows, fields_by_row):
        if len(fields) < 2:
            continue
        counts['two_literal_fields'] += 1
        selected = []
        alternatives = []
        for field in fields:
            if any(field['label'].casefold() == other['label'].casefold()
                   for other in selected):
                continue
            alternate = choose_alternate(rng, field, pool, row['abstract'])
            if alternate is not None:
                selected.append(field)
                alternatives.append(alternate)
            if len(selected) == 2:
                break
        if len(selected) != 2:
            counts['no_two_same_type_alternatives'] += 1
            continue
        try:
            four = make_four_arm(row, selected, alternatives)
        except ValueError:
            counts['semantic_shape_rejected'] += 1
            continue
        counts['valid_four_arm'] += 1
        pairs[tuple(sorted(field['label'].casefold()
                           for field in selected))] += 1
        both_fit = True
        all_full_fit = True
        for arm in (four.original, four.edit_first,
                    four.edit_second, four.edit_both):
            enc = tokenizer.encode(arm.text)
            all_full_fit &= len(enc.ids) <= 256
            for site in arm.sites:
                lo, hi = site.span
                ids = [i for i, (a, b) in enumerate(enc.offsets)
                       if a < hi and b > lo]
                if not ids:
                    counts['unaligned_target'] += 1
                    both_fit = False
                    continue
                left = arm.text[enc.offsets[ids[0]][0]:lo]
                right = arm.text[hi:enc.offsets[ids[-1]][1]]
                if left.strip() or right.strip():
                    counts['unaligned_target'] += 1
                    both_fit = False
                else:
                    both_fit &= ids[-1] < 256
        counts['all_eight_targets_under_256'] += both_fit
        counts['all_four_full_docs_under_256'] += all_full_fit
        if len(original) < panel_size:
            o = first_words(four.original.text.split('\n\n', 1)[1],
                            form_words)
            e = first_words(four.edit_both.text.split('\n\n', 1)[1],
                            form_words)
            if o and e:
                original.append(o)
                edited.append(e)
        if len(examples) < 10:
            examples.append((row['name'], tuple(x['label'] for x in selected),
                             tuple(x['value'] for x in selected),
                             tuple(alternatives)))
    result = {
        'source': 'wikimedia/structured-wikipedia enwiki shard 0',
        'seed': seed, 'scan_rows': scan_rows,
        'form_words': form_words, 'counts': dict(counts),
        'top_field_pairs': [[list(pair), n]
                            for pair, n in pairs.most_common(30)],
        'form_panel_n': len(original),
        'original_prose_form': structure(original) if original else None,
        'both_edited_prose_form': structure(edited) if edited else None,
        'unselected_wikipedia_form': structure(rng.sample(
            unselected, min(len(unselected), len(original))))
            if unselected and original else None,
        'caveat': 'The paired edits enforce two literal field-to-prose '
                  'relations, not every factual dependency in a Wikipedia '
                  'article. Value swaps can contradict other untracked '
                  'claims; manual semantic review is required before '
                  'training.',
    }
    if natural_preview is not None:
        previews = json.loads(natural_preview.read_text())
        crops = [first_words(x['preview'], form_words) for x in previews]
        crops = [crop for crop in crops if crop]
        reference = rng.sample(crops, min(len(crops), len(original)))
        result['natural_reference_form'] = structure(reference)
    return result, examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parquet', type=Path, required=True)
    parser.add_argument('--scan-rows', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=74451)
    parser.add_argument('--panel-size', type=int, default=512)
    parser.add_argument('--form-words', type=int, default=40)
    parser.add_argument('--natural-preview', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result, examples = audit(args.parquet, scan_rows=args.scan_rows,
                             seed=args.seed, panel_size=args.panel_size,
                             form_words=args.form_words,
                             natural_preview=args.natural_preview)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(result['counts'], flush=True)
    for row in examples[:5]:
        print(row, flush=True)


if __name__ == '__main__':
    main()

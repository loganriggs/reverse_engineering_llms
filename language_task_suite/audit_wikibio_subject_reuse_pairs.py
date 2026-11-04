"""Audit strict empirical biography subject-reuse counterfactual pairs."""

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
from audit_wikibio_document_rules import candidate, first_words
from wikibio_subject_reuse_dgp import make_pair


def target_tokens(tokenizer, arm):
    encoding = tokenizer.encode(arm.text)
    lo, hi = arm.target_span
    ids = [i for i, (a, b) in enumerate(encoding.offsets)
           if a < hi and b > lo]
    if not ids or encoding.offsets[ids[0]][0] != lo or \
            encoding.offsets[ids[-1]][1] != hi:
        raise ValueError('Target does not align to tokenizer offsets')
    return ids, len(encoding.ids)


def audit(parquet, *, scan_rows=20000, seed=74351,
          panel_size=512, form_words=40, natural_preview=None):
    rng = random.Random(seed)
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    eligible = []
    same_corpus_form_pool = []
    scanned = 0
    file = pq.ParquetFile(parquet)
    for batch in file.iter_batches(batch_size=1024):
        for row in batch.to_pylist():
            if scanned >= scan_rows:
                break
            scanned += 1
            crop = first_words(row['target_text'], form_words)
            if crop is not None:
                same_corpus_form_pool.append(crop)
            site = candidate(row)
            if site is not None and site['strict_editable']:
                eligible.append((row, site['surname']))
        if scanned >= scan_rows:
            break
    counts = Counter(scanned=scanned, strict_candidates=len(eligible))
    surnames = [surname for _, surname in eligible]
    rng.shuffle(eligible)
    original, edited = [], []
    target_lengths, site_positions = [], []
    for row, surname in eligible:
        for _ in range(20):
            alternative = rng.choice(surnames)
            try:
                pair = make_pair(row, alternative)
            except ValueError:
                continue
            break
        else:
            counts['pair_rejected'] += 1
            continue
        counts['valid_pairs'] += 1
        orig_ids, orig_length = target_tokens(tokenizer, pair.original)
        edit_ids, edit_length = target_tokens(tokenizer, pair.edited)
        counts['both_target_starts_under_256'] += (
            orig_ids[0] < 256 and edit_ids[0] < 256)
        counts['both_target_ends_under_256'] += (
            orig_ids[-1] < 256 and edit_ids[-1] < 256)
        counts['both_full_docs_under_256'] += (
            orig_length <= 256 and edit_length <= 256)
        target_lengths.extend((len(orig_ids), len(edit_ids)))
        site_positions.extend((orig_ids[0], edit_ids[0]))
        if len(original) < panel_size:
            o = first_words(pair.original.text, form_words)
            e = first_words(pair.edited.text, form_words)
            if o and e:
                original.append(o)
                edited.append(e)
    result = {
        'source': 'thoughtworks/wiki_bio train parquet',
        'seed': seed, 'scan_rows': scan_rows,
        'form_words': form_words,
        'counts': dict(counts),
        'median_target_bpe_tokens': sorted(target_lengths)[
            len(target_lengths)//2] if target_lengths else None,
        'median_target_start_bpe': sorted(site_positions)[
            len(site_positions)//2] if site_positions else None,
        'form_panel_size': len(original),
        'original_form': structure(original) if original else None,
        'edited_form': structure(edited) if edited else None,
        'unselected_wikibio_form': structure(rng.sample(
            same_corpus_form_pool,
            min(len(same_corpus_form_pool), len(original))))
            if same_corpus_form_pool and original else None,
        'caveat': 'This verifies only a visible subject-name reuse rule. '
                  'The biography can contain other facts not in its '
                  'infobox; a consistent surname edit is an alternate '
                  'world, not a natural Wikipedia article. No text is '
                  'exported in this aggregate.',
    }
    if natural_preview is not None:
        rows = json.loads(natural_preview.read_text())
        crops = [first_words(row['preview'], form_words) for row in rows]
        crops = [crop for crop in crops if crop]
        natural = rng.sample(crops, min(len(crops), len(original)))
        result['natural_reference_n'] = len(natural)
        result['natural_reference_form'] = structure(natural)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parquet', type=Path, required=True)
    parser.add_argument('--scan-rows', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=74351)
    parser.add_argument('--panel-size', type=int, default=512)
    parser.add_argument('--form-words', type=int, default=40)
    parser.add_argument('--natural-preview', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.parquet, scan_rows=args.scan_rows,
                   seed=args.seed, panel_size=args.panel_size,
                   form_words=args.form_words,
                   natural_preview=args.natural_preview)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(result['counts'], flush=True)


if __name__ == '__main__':
    main()

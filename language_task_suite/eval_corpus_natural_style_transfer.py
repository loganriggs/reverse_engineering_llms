"""Evaluate natural US/UK spelling cues with source-only interventions."""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import islice
import json
import os
from pathlib import Path
import random
import sys

from datasets import load_dataset
import torch
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from corpus_rule_reference_dgp import Document, RuleSite, chunk_document, sample_document
from corpus_transformed_spelling_style import LOOKUP
from train_corpus_delimiter_bilinear import _model


def _only_local(document, tokenizer, n_ctx):
    chunks = chunk_document(document, tokenizer, n_ctx)
    matches = [(chunk, site) for chunk in chunks for site in chunk['sites']
               if site['kind'] == 'natural_style']
    return matches[0] if len(matches) == 1 else None


def pair_for_site(document, site, tokenizer, n_ctx):
    """Return original/source-swapped chunks with the same target token."""
    original = Document(document.source_text, document.text, (site,),
                        ('natural_style',), ('natural_style',))
    first = _only_local(original, tokenizer, n_ctx)
    if first is None:
        return None
    cue = document.text[slice(*site.source)]
    root, us, uk = LOOKUP[cue]
    alternate = uk if cue == us else us
    start, end = site.source
    delta = len(alternate) - (end - start)
    edited_text = document.text[:start] + alternate + document.text[end:]
    edited_site = RuleSite('natural_style',
                           (start, start + len(alternate)),
                           (site.target[0] + delta, site.target[1] + delta),
                           site.answer)
    edited = Document(document.source_text, edited_text, (edited_site,),
                      ('natural_style',), ('natural_style',))
    second = _only_local(edited, tokenizer, n_ctx)
    if second is None:
        return None
    original_chunk, original_site = first
    edited_chunk, edited_local_site = second
    actual_id = original_site['target_ids'][0]
    if edited_local_site['target_ids'] != (actual_id,):
        return None
    target_root, target_us, target_uk = LOOKUP[site.answer]
    assert root != target_root
    target_is_us = site.answer == target_us
    other_word = target_uk if target_is_us else target_us
    encoded = tokenizer.id_to_token(actual_id)
    prefix = 'Ġ' if encoded.startswith('Ġ') else ''
    if encoded != prefix + site.answer:
        return None
    other_id = tokenizer.token_to_id(prefix + other_word)
    if other_id is None or other_id == actual_id:
        return None
    return {'x': original_chunk['x'], 'y': original_chunk['y'],
            'position': original_site['positions'][0],
            'edited_x': edited_chunk['x'],
            'edited_position': edited_local_site['positions'][0],
            'actual_id': actual_id, 'other_id': other_id,
            'cue_us': cue == us, 'target_us': target_is_us,
            'cue_root': root, 'target_root': target_root}


def build_panel(start, n, seed, tokenizer, n_ctx):
    counts, rows = Counter(), []
    dataset = load_dataset('HuggingFaceFW/fineweb', name='sample-10BT',
                           split='train', streaming=True)
    stream = iter(dataset)
    try:
        for offset, datum in enumerate(islice(stream, start, start + n), start):
            counts['sources'] += 1
            if counts['sources'] % 1000 == 0:
                print('scanned', counts['sources'], flush=True)
            document = sample_document(
                datum['text'], random.Random(seed + offset), (),
                list_rate=0., bracket_rate=0., nonce_rate=0.,
                literal_copy_rate=0., natural_style_rate=1.)
            if 'natural_style' not in document.applied:
                continue
            counts['sources_with_style_pair'] += 1
            # One target per source keeps document-level metrics independent.
            for site in document.sites:
                row = pair_for_site(document, site, tokenizer, n_ctx)
                if row is not None:
                    rows.append(row)
                    counts['sources_with_valid_pair'] += 1
                    counts['matching_cue_target'] += (
                        row['cue_us'] == row['target_us'])
                    target_variant = 'us' if row['target_us'] else 'uk'
                    counts[f'target_{target_variant}'] += 1
                    counts[f'target_root_{row["target_root"]}_{target_variant}'] += 1
                    break
    finally:
        if hasattr(stream, 'close'):
            stream.close()
    assert counts['sources'] == n
    return rows, dict(counts)


@torch.no_grad()
def score(model, rows, device, batch_size):
    model.eval()
    tally = Counter()
    for start in range(0, len(rows), batch_size):
        part = rows[start:start + batch_size]
        x = torch.stack([r['x'] for r in part]).to(device)
        edited_x = torch.stack([r['edited_x'] for r in part]).to(device)
        original = model(x).float()
        edited = model(edited_x).float()
        for j, row in enumerate(part):
            a, b = row['actual_id'], row['other_id']
            position, edited_position = row['position'], row['edited_position']
            orig_logits = original[j, position]
            edit_logits = edited[j, edited_position]
            original_margin = float(orig_logits[a] - orig_logits[b])
            edited_margin = float(edit_logits[a] - edit_logits[b])
            # Positive means changing the cue moves probability toward the
            # newly implied spelling, irrespective of the observed target.
            cue_response = original_margin - edited_margin
            if row['cue_us'] != row['target_us']:
                cue_response = -cue_response
            group = 'agree' if row['cue_us'] == row['target_us'] else 'disagree'
            for key in ('all', group):
                tally[f'{key}_n'] += 1
                tally[f'{key}_target_two_way'] += original_margin > 0
                tally[f'{key}_target_top1'] += (
                    int(orig_logits.argmax()) == a)
                tally[f'{key}_cue_response_positive'] += cue_response > 0
                tally[f'{key}_cue_response_sum'] += cue_response
                tally[f'{key}_actual_margin_sum'] += original_margin
    result = {}
    for group in ('all', 'agree', 'disagree'):
        n = tally[f'{group}_n']
        result[group] = {'n': n,
                         'target_two_way': tally[f'{group}_target_two_way']/n
                         if n else None,
                         'target_top1': tally[f'{group}_target_top1']/n
                         if n else None,
                         'positive_cue_response':
                         tally[f'{group}_cue_response_positive']/n
                         if n else None,
                         'mean_cue_response_logit':
                         tally[f'{group}_cue_response_sum']/n if n else None,
                         'mean_actual_target_margin':
                         tally[f'{group}_actual_margin_sum']/n if n else None}
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--start', type=int, default=423000)
    p.add_argument('--n', type=int, default=5000)
    p.add_argument('--seed', type=int, default=85501)
    p.add_argument('--n-ctx', type=int, default=256)
    p.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available()
                   else 'cpu')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--pair-cache', type=Path,
                   help='Local token-only cache for repeat model evaluations')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    identity = (args.start, args.n, args.seed, args.n_ctx)
    if args.pair_cache and args.pair_cache.exists():
        cached = torch.load(args.pair_cache, map_location='cpu',
                            weights_only=True)
        if tuple(cached['identity']) != identity:
            raise ValueError('Pair cache identity mismatch')
        rows, counts = cached['rows'], cached['counts']
    else:
        rows, counts = build_panel(args.start, args.n, args.seed,
                                   tokenizer, args.n_ctx)
        if args.pair_cache:
            args.pair_cache.parent.mkdir(parents=True, exist_ok=True)
            temp = args.pair_cache.with_suffix('.tmp')
            torch.save({'identity': identity, 'rows': rows, 'counts': counts},
                       temp)
            temp.replace(args.pair_cache)
    if not rows:
        raise RuntimeError('No paired natural style sites')
    results = {}
    for checkpoint in args.checkpoints:
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if saved['n_ctx'] != args.n_ctx:
            raise ValueError('Checkpoint/context mismatch')
        model = _model(saved['vocab_size'], saved['n_ctx'], saved['depth'],
                       saved['d_model'], saved['heads'], args.device,
                       saved['norm'])
        model.load_state_dict(saved['state_dict'])
        label = checkpoint.parent.name + '/' + checkpoint.name
        if label in results:
            raise ValueError(f'Duplicate checkpoint label: {label}')
        results[label] = score(model, rows, args.device,
                               args.batch_size)
        print('evaluated', label, flush=True)
        del model
    report = {'config': {k: [str(x) for x in v] if isinstance(v, list)
                         else str(v) if isinstance(v, Path) else v
                         for k, v in vars(args).items()},
              'counts': counts, 'results': results,
              'scope': 'Fresh unedited FineWeb source offsets, one valid '
                       'natural style pair per document. Source-only cue '
                       'spelling swap is diagnostic; natural target stays '
                       'fixed. Positive logit response is cue sensitivity, '
                       'not proof of a correct counterfactual data oracle. '
                       'No source prose or model checkpoint saved here.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print('saved', args.output, flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()

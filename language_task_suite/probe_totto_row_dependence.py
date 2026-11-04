"""Paired source-row swap probe for a trained ToTTo bilinear LM.

This tests whether logits at exact value mentions depend on the displayed
table row. It does not certify correct value binding: donor rows usually have
different headers and are an imperfect, mildly OOD intervention.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from audit_totto_table_description_dgp import DEFAULT_TOKENIZER
from train_corpus_delimiter_bilinear import _model
from train_totto_table_description_bilinear import select


def value_line_positions(row, tokenizer):
    ids = row['x'][:row['n_tokens']].tolist()
    text = tokenizer.decode(ids)
    encoding = tokenizer.encode(text)
    if encoding.ids != ids:
        raise ValueError('Token decoding did not round trip')
    lines = text.splitlines(keepends=True)
    table_lines = [line for line in lines if line.startswith('| ')]
    if len(table_lines) != 2:
        raise ValueError('Expected one header and one value line')
    line = table_lines[1]
    lo = text.index(line)
    hi = lo + len(line)
    positions = tuple(i for i, (a, b) in enumerate(encoding.offsets)
                      if a < hi and b > lo)
    if not positions or positions[-1] >= min(p[0] for p in row['sites']):
        raise ValueError('Value line overlaps target')
    return positions


@torch.no_grad()
def score(model, row, modified_x, device):
    x = modified_x.long().unsqueeze(0).to(device)
    y = row['y'].long().to(device)
    logits = model(x)[0].float()
    valid = y != -100
    per = F.cross_entropy(logits[valid], y[valid], reduction='none')
    per_full = torch.zeros_like(y, dtype=torch.float32)
    per_full[valid] = per
    positions = sorted({p for site in row['sites'] for p in site})
    first = [site[0] for site in row['sites']]
    desc_start = min(positions)
    background = [p for p in range(desc_start, row['n_tokens'])
                  if p not in positions]
    return {'site_nll': per_full[positions].mean().item(),
            'first_nll': per_full[first].mean().item(),
            'background_nll': per_full[background].mean().item()
            if background else None,
            'first_exact': sum((logits[p].argmax() == y[p]).item()
                               for p in first) / len(first)}


def probe(cache_path, state_path, *, split='test', n=128, seed=74901,
          device='cuda'):
    cache = torch.load(cache_path, map_location='cpu', weights_only=True)
    state = torch.load(state_path, map_location='cpu', weights_only=True)
    identity = state['identity']
    model = _model(cache['vocab_size'], cache['n_ctx'], identity[1],
                   identity[2], identity[3], device, 'rms')
    model.load_state_dict(state['model'])
    model.eval()
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    rows = cache['splits'][split]
    eval_seed = seed + 1000 + ('val', 'test', 'combo_val',
                               'combo_test').index(split)
    selected = select(rows, n, eval_seed)
    donors = defaultdict(list)
    for j, row in enumerate(rows):
        positions = value_line_positions(row, tokenizer)
        donors[len(positions)].append((j, positions))
    rng = random.Random(seed + 4567)
    records = []
    for row in selected:
        positions = value_line_positions(row, tokenizer)
        candidates = donors[len(positions)]
        if len(candidates) < 2:
            continue
        other_index, other_positions = rng.choice(candidates)
        donor = rows[other_index]
        # Reject identical value lines; the intervention must change tokens.
        if torch.equal(row['x'][list(positions)],
                       donor['x'][list(other_positions)]):
            continue
        changed = row['x'].clone()
        changed[list(positions)] = donor['x'][list(other_positions)]
        original = score(model, row, row['x'], device)
        swapped = score(model, row, changed, device)
        records.append({'original': original, 'swapped': swapped,
                        'site_count': len(row['sites']),
                        'row_tokens': len(positions)})
    if not records:
        raise ValueError('No same-length row swaps')
    summary = {'n': len(records), 'split': split,
               'source': str(cache_path), 'state': str(state_path),
               'checkpoint_step': state['step'],
               'intervention': 'Replace all token IDs in the displayed value '
                               'row with another heldout row of the same '
                               'token length; leave target sentence fixed.',
               'caveat': 'This is source-row sensitivity, not a valid '
                         'alternate-world value-binding test. Most donor '
                         'rows have different schemas.'}
    for metric in ('site_nll', 'first_nll', 'background_nll', 'first_exact'):
        pairs = [(record['original'][metric], record['swapped'][metric])
                 for record in records
                 if record['original'][metric] is not None]
        summary[metric] = {
            'original_mean': sum(a for a, _ in pairs) / len(pairs),
            'swapped_mean': sum(b for _, b in pairs) / len(pairs),
            'mean_delta_swapped_minus_original': sum(b-a for a, b in pairs)
                                                 / len(pairs),
            'fraction_swap_worse': sum(b > a for a, b in pairs) / len(pairs)}
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=('val','test','combo_val',
                                            'combo_test'), default='test')
    parser.add_argument('--n', type=int, default=128)
    parser.add_argument('--seed', type=int, default=74901)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = probe(args.cache, args.state, split=args.split,
                   n=args.n, seed=args.seed, device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

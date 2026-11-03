"""Diagnostic: add the opposite style for each training prefix.

The resulting cache stays outside Git. This deliberately repeats the same
source document in both styles, so it is an augmentation control rather than
an independent natural-language sample.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from corpus_transformed_spelling_style import FAMILIES


def token_flip_map(tokenizer):
    result = {}
    for pairs in FAMILIES.values():
        for us, uk in pairs:
            for prefix in ('', ' '):
                left = tokenizer.encode(prefix+us,
                                        add_special_tokens=False).ids
                right = tokenizer.encode(prefix+uk,
                                         add_special_tokens=False).ids
                if len(left) == len(right) == 1:
                    for src, dst in ((left[0], right[0]),
                                     (right[0], left[0])):
                        assert src not in result or result[src] == dst
                        result[src] = dst
    assert all(result[result[x]] == x for x in result)
    return result


def augment(data):
    mapping = token_flip_map(Tokenizer.from_file(str(DEFAULT_TOKENIZER)))
    row = data['train']
    n, width = row['x'].shape
    original = {key: value.clone() for key, value in row.items()}
    flipped = {key: value.clone() for key, value in row.items()}
    arange = torch.arange(n)
    cue = row['cue_position']
    site = row['site_position']
    assert bool((cue <= site).all())
    assert bool((site < width).all())
    for idx in range(n):
        c = int(cue[idx]); s = int(site[idx])
        cue_id = int(row['x'][idx, c]); target_id = int(row['target_id'][idx])
        assert cue_id in mapping and target_id in mapping
        original['x'][idx, s+1:] = 50256
        original['y'][idx, s+1:] = -100
        flipped['x'][idx, c] = mapping[cue_id]
        if c:
            flipped['y'][idx, c-1] = mapping[cue_id]
        flipped['y'][idx, s] = mapping[target_id]
        flipped['target_id'][idx] = mapping[target_id]
        flipped['x'][idx, s+1:] = 50256
        flipped['y'][idx, s+1:] = -100
        assert original['y'][idx, s] == original['target_id'][idx]
        assert flipped['y'][idx, s] == flipped['target_id'][idx]
        assert (original['x'][idx, :s+1] !=
                flipped['x'][idx, :s+1]).nonzero().flatten().tolist() == [c]
        assert (original['y'][idx, :s+1] !=
                flipped['y'][idx, :s+1]).nonzero().flatten().tolist() == (
                    [c-1, s] if c else [s])
    data['train'] = {key: torch.cat((original[key], flipped[key]))
                     for key in row}
    assert len(data['train']['x']) == 2*n
    return data


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    data = torch.load(args.source, weights_only=True)
    data = augment(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, args.output)
    print('Saved', len(data['train']['x']), 'paired training windows')


if __name__ == '__main__':
    main()

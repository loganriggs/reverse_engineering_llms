"""Evaluate saved bilinear checkpoints on old and unedited background rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from train_corpus_delimiter_bilinear import _model


@torch.no_grad()
def nll(model, rows, indices, device, batch_size=4):
    total, tokens = 0., 0
    for start in range(0, len(indices), batch_size):
        ix = indices[start:start+batch_size]
        x = rows['x'][ix].to(device)
        y = rows['y'][ix].to(device)
        out = model(x).float()
        total += F.cross_entropy(out.flatten(0, 1), y.flatten(),
                                 ignore_index=-100, reduction='sum').item()
        tokens += int((y != -100).sum())
    return {'rows': len(indices), 'tokens': tokens,
            'token_nll': total/tokens if tokens else None}


def audit(old_path, new_path, checkpoints, device):
    old = torch.load(old_path, map_location='cpu', weights_only=True)
    new = torch.load(new_path, map_location='cpu', weights_only=True)
    result = {'old_cache': str(old_path), 'new_cache': str(new_path),
              'checkpoints': {}, 'cache_checks': {}}
    for split in ('train', 'val', 'test'):
        a, b = old[split], new[split]
        assert a['source_hashes'] == b['source_hashes']
        assert a['background']['x'].shape == b['background']['x'].shape
        changed = (a['background']['x'] != b['background']['x']).any(1)
        for kind in ('bracket', 'literal_copy', 'natural_style', 'list'):
            aa, bb = a['sites'][kind], b['sites'][kind]
            assert (aa is None) == (bb is None)
            if aa is not None:
                assert all(torch.equal(aa[key], bb[key]) for key in aa)
        for aa, bb in zip(a['name_pairs'], b['name_pairs']):
            assert all(torch.equal(aa[key], bb[key]) for key in aa)
        result['cache_checks'][split] = {
            'sources': len(a['source_hashes']),
            'changed_background_rows': int(changed.sum()),
            'changed_background_tokens': int((a['background']['x'] !=
                                              b['background']['x']).sum()),
            'all_task_rows_identical': True}
    torch.set_num_threads(4)
    for name, path in checkpoints.items():
        saved = torch.load(path, map_location='cpu', weights_only=True)
        model = _model(50257, 256, 2, 64, 4, device, 'rms')
        model.load_state_dict(saved['model'])
        model.eval()
        rows = {}
        for split in ('val', 'test'):
            a, b = old[split]['background'], new[split]['background']
            changed = (a['x'] != b['x']).any(1).nonzero().flatten().tolist()
            first = list(range(min(128, len(a['x']))))
            rows[split] = {selection: {
                'old': nll(model, a, indices, device),
                'unedited': nll(model, b, indices, device)}
                for selection, indices in (('first128', first),
                                           ('changed', changed))}
        result['checkpoints'][name] = {'path': str(path),
                                       'step': saved['step'],
                                       'background': rows}
    result['scope'] = ('The old and corrected caches have identical source '
                       'splits and task rows; only the ordinary background '
                       'draw is changed. These checkpoints were trained '
                       'on the old cache. NLL compares different target '
                       'tokens in edited rows, so it is a distribution '
                       'correction diagnostic, not causal-task behavior.')
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--old-cache', type=Path, required=True)
    p.add_argument('--new-cache', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--paired', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = audit(args.old_cache, args.new_cache,
                   {'baseline': args.baseline, 'paired': args.paired},
                   args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(result['cache_checks'])
    for name, row in result['checkpoints'].items():
        print(name, row['background']['test'])


if __name__ == '__main__':
    main()

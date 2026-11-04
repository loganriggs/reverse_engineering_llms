"""Strict all-arm two-rule evaluation of frozen list/bracket checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from train_corpus_delimiter_bilinear import _model
from train_corpus_list_delimiter_mixture import eval_joint


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    torch.set_num_threads(4)
    data = torch.load(a.cache, map_location='cpu', weights_only=True)
    result = {'cache': str(a.cache), 'models': {},
              'scope': 'Joint four-arm source documents are absent from '
                       'training. All-arm two-way requires each of eight '
                       'target-versus-alternative comparisons; top1 '
                       'requires every full-vocabulary prediction.'}
    for path in a.checkpoints:
        saved = torch.load(path, map_location='cpu', weights_only=True)
        model = _model(50257, data['n_ctx'], 2, 64, 4, a.device, 'none')
        model.load_state_dict(saved['state_dict'])
        result['models'][path.stem] = {
            split: eval_joint(model, data['joint'][split], a.device)
            for split in ('val', 'test')}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    for name, row in result['models'].items():
        print(name, row['test'], flush=True)


if __name__ == '__main__':
    main()

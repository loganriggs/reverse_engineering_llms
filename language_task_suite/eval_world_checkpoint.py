"""Re-evaluate a saved document-world checkpoint against a fixed generator version."""

import argparse
import importlib
import json
from pathlib import Path

import torch

from train import DeepModel, SPECS, encode, evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument('checkpoint')
    p.add_argument('--world-version', choices=('v1', 'v2'), required=True)
    p.add_argument('--test-n', type=int, default=2000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    world = importlib.import_module('document_world' if a.world_version == 'v1'
                                    else 'document_world_v2')
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    assert ck['training_config'].get('world_version', 'v1') == a.world_version
    device = torch.device(a.device)
    k = ck['model_kwargs']
    model = DeepModel(len(ck['vocabulary']), k['d_model'], k['n_head'],
                      SPECS[f"block{ck['depth']}"], ck['n_ctx'],
                      norm=k['norm'], residual=k['residual'],
                      attention=k['attention']).to(device)
    model.load_state_dict(ck['state_dict'])
    model.eval()
    specs = [('iid', a.seed+1, 'train', (0, 1)),
             ('composition', a.seed+2, 'composition', (0, 1)),
             ('held_genre', a.seed+3, 'train', (2,))]
    result = {}
    for name, seed, split, genres in specs:
        examples = world.sample_world(a.test_n, seed, split=split, genres=genres)
        tokens = {t for ex in examples for t in ex.prefix + (ex.target,)}
        assert tokens.issubset(ck['vocabulary']), sorted(tokens-set(ck['vocabulary']))
        result[name] = evaluate(model, encode(examples, ck['vocabulary'], device))
    output = dict(config=vars(a), depth=ck['depth'], results=result)
    Path(a.output).write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({name: row['full_accuracy'] for name, row in result.items()}))


if __name__ == '__main__':
    main()

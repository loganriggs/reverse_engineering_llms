"""Evaluate v1 and v2 checkpoints on the same latent worlds.

This controls the particular sampled states and event histories, but models
still differ in training text, random initialization, and optimization path.
"""

import argparse
import json
from pathlib import Path

import torch

import document_world as v1
import document_world_v2 as v2
from train import DeepModel, SPECS, encode, evaluate


def load_model(path, version, device):
    ck = torch.load(path, map_location='cpu', weights_only=True)
    assert ck['training_config'].get('world_version', 'v1') == version
    k = ck['model_kwargs']
    model = DeepModel(len(ck['vocabulary']), k['d_model'], k['n_head'],
                      SPECS[f"block{ck['depth']}"], ck['n_ctx'],
                      norm=k['norm'], residual=k['residual'],
                      attention=k['attention']).to(device)
    model.load_state_dict(ck['state_dict'])
    model.eval()
    return model, ck


def main():
    p = argparse.ArgumentParser()
    p.add_argument('v1_checkpoint')
    p.add_argument('v2_checkpoint')
    p.add_argument('--n', type=int, default=2000)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    device = torch.device(a.device)
    models = {'v1': load_model(a.v1_checkpoint, 'v1', device),
              'v2': load_model(a.v2_checkpoint, 'v2', device)}
    splits = [('iid', 8241, 'train', (0,1)),
              ('composition', 8242, 'composition', (0,1)),
              ('held_genre', 8243, 'train', (2,))]
    rows = {}
    for name, seed, split, genres in splits:
        newer = v2.sample_world(a.n, seed, split=split, genres=genres)
        older = [v1.realize(ex.latent, ex.variant) for ex in newer]
        assert [ex.target for ex in older] == [ex.target for ex in newer]
        rows[name] = {}
        for version, examples in [('v1', older), ('v2', newer)]:
            model, ck = models[version]
            rows[name][version] = evaluate(model, encode(examples, ck['vocabulary'], device))
    result = dict(config=vars(a), caveat='Matched latent test worlds, not matched '
                  'training examples or optimization trajectories.', results=rows)
    Path(a.output).write_text(json.dumps(result, indent=2)+'\n')
    print({split: {version: round(row['full_accuracy'], 4)
                   for version, row in versions.items()}
           for split, versions in rows.items()})


if __name__ == '__main__':
    main()

"""Paired semantic edits for document-world bilinear checkpoints."""

import argparse
import importlib
import json
from collections import defaultdict
from pathlib import Path

import torch

from train import DeepModel, SPECS, encode


@torch.no_grad()
def score(model, examples, vocab, device):
    data = encode(examples, vocab, device)
    x, last, targets = data[:3]
    preds = []
    for start in range(0, len(examples), 128):
        end = min(start+128, len(examples))
        logits = model(x[start:end])
        logits = logits[torch.arange(end-start, device=device), last[start:end]]
        preds.extend(logits.argmax(-1).tolist())
    correct = [preds[i] == targets[i].item() for i in range(len(examples))]
    return dict(pair_accuracy=sum(correct[i] and correct[i+1]
                                  for i in range(0, len(correct), 2))/(len(correct)//2),
                original_accuracy=sum(correct[::2])/(len(correct)//2),
                edited_accuracy=sum(correct[1::2])/(len(correct)//2),
                prediction_invariance=sum(preds[i] == preds[i+1]
                                          for i in range(0, len(preds), 2))/(len(preds)//2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('checkpoint')
    p.add_argument('--n', type=int, default=300)
    p.add_argument('--world-version', choices=('v1', 'v2'), default='v1')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output', default=None)
    a = p.parse_args()
    world = importlib.import_module('document_world' if a.world_version == 'v1'
                                    else 'document_world_v2')
    causal_pair, nuisance_pair, sample_world = (
        world.causal_pair, world.nuisance_pair, world.sample_world)
    torch.set_num_threads(4)
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    k = ck['model_kwargs']
    device = torch.device(a.device)
    model = DeepModel(len(ck['vocabulary']), k['d_model'], k['n_head'],
                      SPECS[f"block{ck['depth']}"], ck['n_ctx'],
                      norm=k['norm'], residual=k['residual'], attention=k['attention']).to(device)
    model.load_state_dict(ck['state_dict'])
    model.eval()
    result = {}
    for split, seed, genres in [('iid', 1043, (0, 1)),
                                ('composition', 1044, (0, 1)),
                                ('held_genre', 1045, (2,))]:
        examples = sample_world(a.n*5, seed, split='composition' if split == 'composition' else 'train',
                                genres=genres)
        grouped = defaultdict(list)
        for ex in examples:
            grouped[ex.family].append(ex)
        result[split] = {}
        for family, group in grouped.items():
            group = group[:a.n]
            result[split][family] = {}
            for kind, transform in [('causal', causal_pair), ('nuisance', nuisance_pair)]:
                pairs = [transform(ex) for ex in group]
                flat = [ex for pair in pairs for ex in pair]
                result[split][family][kind] = score(model, flat, ck['vocabulary'], device)
    output = dict(checkpoint=a.checkpoint, world_version=a.world_version,
                  n_target_pairs=a.n, results=result)
    print(json.dumps(output, indent=2))
    if a.output:
        Path(a.output).write_text(json.dumps(output, indent=2)+'\n')


if __name__ == '__main__':
    main()

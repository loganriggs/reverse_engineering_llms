"""Evaluate answer consistency under controlled causal and nuisance edits."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import replace
from pathlib import Path

import torch

from grammar import COLORS, PLACES, generate
from train import DeepModel, SPECS, encode


def edited(ex, prefix, target):
    return replace(ex, prefix=tuple(prefix), target=target)


def pairs(n: int, seed: int, variant: int):
    rng = random.Random(seed)
    out = {}
    for family in ('attribute_binding', 'conditional_rule', 'state_update'):
        causal, nuisance = [], []
        for _ in range(n):
            ex = generate(family, rng, variant)
            l = dict(ex.latent)
            if family == 'attribute_binding':
                other = l['second'] if l['query'] == l['first'] else l['first']
                answer = l['second_color'] if other == l['second'] else l['first_color']
                prefix = list(ex.prefix)
                prefix[-3 if variant else -2] = other
                causal.append((ex, edited(ex, prefix, answer)))
                # Change the color of the unqueried person's folder only.
                spare = next(c for c in COLORS if c not in (l['first_color'], l['second_color']))
                prefix = list(ex.prefix)
                old_color = l['second_color'] if l['query'] == l['first'] else l['first_color']
                position = prefix.index(old_color)
                prefix[position] = spare
                nuisance.append((ex, edited(ex, prefix, ex.target)))
            elif family == 'conditional_rule':
                rules = dict(pair.split(':') for pair in l['rules'].split(';'))
                other = next(c for c in rules if c != l['color'])
                prefix = list(ex.prefix)
                prefix[-6] = other
                causal.append((ex, edited(ex, prefix, rules[other])))
                prefix = list(ex.prefix)
                # Change only the nonqueried rule's lexical color key, preserving its state.
                position = prefix.index(other)
                prefix[position] = next(c for c in COLORS if c not in rules)
                nuisance.append((ex, edited(ex, prefix, ex.target)))
            else:
                obj, other = l['object'], l['distractor_object']
                prefix = list(ex.prefix)
                move = prefix.index('moved')
                new = next(p for p in PLACES if p not in (l['old'], l['new']))
                prefix[move+5] = new
                causal.append((ex, edited(ex, prefix, new)))
                prefix = list(ex.prefix)
                other_index = prefix.index(other)
                prefix[other_index+3] = next(p for p in PLACES if p != prefix[other_index+3])
                nuisance.append((ex, edited(ex, prefix, ex.target)))
        out[family] = dict(causal=causal, nuisance=nuisance)
    return out


@torch.no_grad()
def score_pairs(model, vocab, cases, device):
    examples = [ex for pair in cases for ex in pair]
    encoded = encode(examples, vocab, device)
    x, last, y = encoded[:3]
    preds = []
    for start in range(0, len(y), 128):
        end = min(start + 128, len(y))
        scores = model(x[start:end])
        preds.extend(scores[torch.arange(end-start, device=device), last[start:end]].argmax(-1).tolist())
    correct = [preds[i] == y[i].item() for i in range(len(y))]
    return dict(pair_accuracy=sum(correct[i] and correct[i+1] for i in range(0, len(y), 2))/len(cases),
                first_accuracy=sum(correct[::2])/len(cases),
                edited_accuracy=sum(correct[1::2])/len(cases),
                prediction_invariance=sum(preds[i] == preds[i+1] for i in range(0, len(y), 2))/len(cases))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('checkpoint')
    p.add_argument('--n', type=int, default=500)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output', default=None)
    a = p.parse_args()
    torch.set_num_threads(4)
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    device = torch.device(a.device)
    k = ck['model_kwargs']
    model = DeepModel(len(ck['vocabulary']), k['d_model'], k['n_head'],
                      SPECS[f"block{ck['depth']}"], ck['n_ctx'],
                      norm=k['norm'], residual=k['residual'], attention=k['attention']).to(device)
    model.load_state_dict(ck['state_dict'])
    model.eval()
    output = {}
    trained_families = set(ck['training_config']['families'].split(','))
    for variant in (0, 1):
        output[str(variant)] = {}
        for family, tests in pairs(a.n, 103+variant, variant).items():
            if family not in trained_families:
                continue
            output[str(variant)][family] = {}
            for kind, cases in tests.items():
                output[str(variant)][family][kind] = score_pairs(model, ck['vocabulary'], cases, device)
    output = dict(checkpoint=a.checkpoint, n_pairs_per_cell=a.n, results=output)
    print(json.dumps(output, indent=2))
    if a.output:
        Path(a.output).write_text(json.dumps(output, indent=2)+'\n')


if __name__ == '__main__':
    main()

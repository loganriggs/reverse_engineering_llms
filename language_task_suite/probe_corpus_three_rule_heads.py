"""Task-blind head-weight ranking versus three heldout rule ablations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from train_corpus_delimiter_bilinear import _model, evaluate as eval_bracket
from train_corpus_list_delimiter_mixture import eval_list
from eval_corpus_nonce_copy import evaluate as eval_nonce


@torch.no_grad()
def weight_scores(model):
    scores = []
    for layer_index, layer in enumerate(model.layers):
        if not hasattr(layer, 'q1'):
            continue
        for head in range(layer.n_head):
            sl = slice(head * layer.d_head, (head + 1) * layer.d_head)
            qk1 = layer.q1.weight[sl].T @ layer.k1.weight[sl]
            qk2 = layer.q2.weight[sl].T @ layer.k2.weight[sl]
            ov = layer.o.weight[:, sl] @ layer.v.weight[sl]
            scores.append({'layer': layer_index, 'head': head,
                           'qk1_norm': qk1.norm().item(),
                           'qk2_norm': qk2.norm().item(),
                           'ov_norm': ov.norm().item(),
                           'product_norm': (qk1.norm() * qk2.norm() *
                                            ov.norm()).item()})
    return scores


def first(rows, n):
    if isinstance(rows, dict):
        return {key: first(value, n) for key, value in rows.items()}
    if isinstance(rows, (list, tuple)):
        return type(rows)(first(value, n) for value in rows)
    return rows[:n]


def measures(model, nonce, lists, brackets, device):
    return {'nonce': eval_nonce(model, nonce, device),
            'list': eval_list(model, lists, device),
            'bracket': eval_bracket(model, brackets, device, batch_size=4)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--nonce-cache', type=Path, required=True)
    p.add_argument('--list-cache', type=Path, required=True)
    p.add_argument('--bracket-cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--n', type=int, default=64)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    model = _model(saved['vocab_size'], saved['n_ctx'], saved['depth'],
                   saved['d_model'], saved['heads'], a.device, saved['norm'])
    model.load_state_dict(saved['state_dict'])
    model.eval()
    nonce_cache = torch.load(a.nonce_cache, map_location='cpu', weights_only=True)
    list_cache = torch.load(a.list_cache, map_location='cpu', weights_only=True)
    bracket_cache = torch.load(a.bracket_cache, map_location='cpu', weights_only=True)
    panels = (first(nonce_cache['paired']['test'], a.n),
              first(list_cache['list']['test'], a.n),
              first(bracket_cache['test'], a.n))
    native = measures(model, *panels, a.device)
    rows = weight_scores(model)
    for row in rows:
        layer = model.layers[row['layer']]
        head, width = row['head'], layer.d_head

        def zero_head(module, inputs, *, head=head, width=width):
            values = inputs[0].clone()
            values[..., head * width:(head + 1) * width] = 0
            return (values,)

        hook = layer.o.register_forward_pre_hook(zero_head)
        try:
            altered = measures(model, *panels, a.device)
        finally:
            hook.remove()
        row['ablated'] = {
            'nonce_greedy_both_drop': native['nonce']['greedy_both'] -
                                      altered['nonce']['greedy_both'],
            'nonce_preference_drop': native['nonce']['both_sequence_preference'] -
                                     altered['nonce']['both_sequence_preference'],
            'list_paired_top1_drop': native['list']['both_top1'] -
                                     altered['list']['both_top1'],
            'bracket_paired_two_way_drop': native['bracket']['both_two_way'] -
                                           altered['bracket']['both_two_way'],
        }
    result = {'config': {key: str(value) if isinstance(value, Path) else value
                         for key, value in vars(a).items()},
              'checkpoint_sha256': hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
              'native': native, 'heads': rows,
              'scope': 'Single-head necessity probe on first n heldout source '
                       'documents per rule. The product norm sees weights '
                       'only; labels enter solely after the ranking is frozen. '
                       'List and bracket have a four-arm joint panel elsewhere; '
                       'nonce is not crossed with them in these documents. '
                       'Ablation is not a complete or unique circuit proof.'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print('saved', a.output, flush=True)
    for row in sorted(rows, key=lambda value: -value['product_norm']):
        print(row['layer'], row['head'],
              round(row['product_norm'], 3), row['ablated'], flush=True)


if __name__ == '__main__':
    main()

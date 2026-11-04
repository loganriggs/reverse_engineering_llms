"""Exact OV gauge selected by weight L1, then matched-coordinate pruning.

The full-rank orthogonal rotation preserves each head's O@V map. Pruning
coordinates changes it. This is a weight-only method; no task labels or
activations enter the gauge fit.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import random

import torch
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from eval_sports_ov_canonical_truncation import (
    max_probe_difference, model_from_checkpoint, replace_ov, teacher_kl,
)
from train_sports_alias_bracket import draw, evaluate, stack


def sparse_rotation(o, v, *, steps=250, lr=.03):
    """Fit a special-orthogonal internal gauge for balanced SVD factors."""
    d = v.shape[0]
    skew_parameter = torch.nn.Parameter(torch.zeros((d, d),
                                                     dtype=torch.float64))
    optimizer = torch.optim.Adam([skew_parameter], lr=lr)
    best_cost = float('inf')
    best_rotation = torch.eye(d, dtype=torch.float64)
    for _ in range(steps):
        rotation = torch.matrix_exp(skew_parameter-skew_parameter.T)
        new_o, new_v = o @ rotation.T, rotation @ v
        # Smooth absolute value avoids a derivative singularity at zero.
        loss = torch.sqrt(new_o.square()+1e-12).sum() + \
               torch.sqrt(new_v.square()+1e-12).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if loss.item() < best_cost:
            best_cost = loss.item()
            best_rotation = rotation.detach().clone()
    return best_rotation, best_cost


def fit_heads(model, *, steps=250, lr=.03):
    """Return CPU factors for every head, with no model input data."""
    rows = []
    with torch.no_grad():
        for layer_index, layer in enumerate(
                layer for layer in model.layers if hasattr(layer, 'q1')):
            for head in range(layer.n_head):
                sl = slice(head*layer.d_head, (head+1)*layer.d_head)
                original_o = layer.o.weight[:, sl].detach().to(
                    'cpu', torch.float64)
                original_v = layer.v.weight[sl, :].detach().to(
                    'cpu', torch.float64)
                product = original_o @ original_v
                u, s, vh = torch.linalg.svd(product, full_matrices=False)
                h = layer.d_head
                o = u[:, :h]*s[:h].sqrt()
                v = s[:h, None].sqrt()*vh[:h]
                original_l1 = original_o.abs().sum()+original_v.abs().sum()
                balanced_l1 = o.abs().sum()+v.abs().sum()
                rows.append({'layer': layer_index, 'head': head,
                             'o': o, 'v': v,
                             'original_l1': float(original_l1),
                             'balanced_l1': float(balanced_l1)})
    for row in rows:
        rotation, cost = sparse_rotation(row['o'], row['v'],
                                         steps=steps, lr=lr)
        row['o'] = row['o'] @ rotation.T
        row['v'] = rotation @ row['v']
        row['sparse_l1'] = cost
        row['orthogonality_error'] = float(torch.linalg.norm(
            rotation @ rotation.T - torch.eye(len(rotation))))
        # This is a numerical identity check on the *product*, independent
        # of a small heldout model probe.
        layer = [item for item in model.layers if hasattr(item, 'q1')][
            row['layer']]
        sl = slice(row['head']*layer.d_head,
                   (row['head']+1)*layer.d_head)
        reference = layer.o.weight[:, sl].detach().to('cpu',
                                                      torch.float64) @ \
                    layer.v.weight[sl, :].detach().to('cpu', torch.float64)
        row['product_relative_error'] = float(torch.linalg.norm(
            row['o'] @ row['v'] - reference)/torch.linalg.norm(reference))
    return rows


@torch.no_grad()
def apply_factors(model, fitted, rank):
    attentions = [layer for layer in model.layers if hasattr(layer, 'q1')]
    for row in fitted:
        layer = attentions[row['layer']]
        h = layer.d_head
        sl = slice(row['head']*h, (row['head']+1)*h)
        o, v = row['o'].clone(), row['v'].clone()
        importance = torch.linalg.norm(o, dim=0)*torch.linalg.norm(v, dim=1)
        keep = set(torch.topk(importance, rank).indices.tolist())
        for index in range(h):
            if index not in keep:
                o[:, index] = 0
                v[index, :] = 0
        layer.o.weight[:, sl].copy_(o.to(layer.o.weight))
        layer.v.weight[sl, :].copy_(v.to(layer.v.weight))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--test-worlds', type=int, default=40)
    parser.add_argument('--seed', type=int, default=41601)
    parser.add_argument('--fit-steps', type=int, default=250)
    parser.add_argument('--fit-lr', type=float, default=.03)
    parser.add_argument('--ranks', default='4,8,16')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(4)
    model, cfg = model_from_checkpoint(args.checkpoint, args.device)
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    rng = random.Random(args.seed)
    # Match the original OV study's test draw: consume its fit worlds first.
    fit, _ = draw(rng, 80, 'train', tokenizer, cfg['n_ctx'], unique=True)
    test, attempts = draw(rng, args.test_worlds, 'heldout', tokenizer,
                          cfg['n_ctx'], unique=True)
    assert not {key for key, _ in fit} & {key for key, _ in test}
    panel = stack(test)
    fitted = fit_heads(model, steps=args.fit_steps, lr=args.fit_lr)
    identity_grams = [torch.eye(cfg['d_model'], dtype=torch.float64)
                      for layer in model.layers if hasattr(layer, 'q1')]
    result = {
        'config': {k: str(v) if isinstance(v, Path) else v
                   for k, v in vars(args).items()},
        'test_attempts': attempts,
        'fit_heads': [{k: v for k, v in row.items()
                       if k not in ('o', 'v')} for row in fitted],
        'native': evaluate(model, panel, args.device, batch_size=4),
        'rows': [],
        'scope': 'Orthogonal gauge rotation of each attention head OV '
                 'factorization minimizes weight L1 at exact full rank. '
                 'Prune whole value coordinates at matched 4/8/16 rank; '
                 'compare with weight SVD on the same heldout games. '
                 'No activation or task labels enter the gauge fit.',
        'caveat': 'This is local OV reduction, not a canonical full '
                  'attention decomposition or a human-legible circuit.'}
    for rank in (int(value) for value in args.ranks.split(',')):
        for method in ('sparse_gauge', 'weight_svd'):
            candidate = deepcopy(model)
            if method == 'sparse_gauge':
                apply_factors(candidate, fitted, rank)
            else:
                replace_ov(candidate, identity_grams, rank, 'weight', .001)
            difference = max_probe_difference(model, candidate, panel,
                                              args.device)
            first_attention = next(layer for layer in model.layers
                                   if hasattr(layer, 'q1'))
            if rank == first_attention.d_head and difference > .001:
                raise AssertionError(('Full rank changed function',
                                      method, difference))
            row = {'method': method, 'rank_per_head': rank,
                   'max_probe_logit_change': difference,
                   'heldout': evaluate(candidate, panel, args.device,
                                      batch_size=4),
                   'teacher_kl': teacher_kl(model, candidate, panel,
                                            args.device)}
            result['rows'].append(row)
            print(method, rank,
                  row['heldout']['bracket']['paired_top1'],
                  row['heldout']['alias']['paired_top1'], flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('saved', args.output)


if __name__ == '__main__':
    main()

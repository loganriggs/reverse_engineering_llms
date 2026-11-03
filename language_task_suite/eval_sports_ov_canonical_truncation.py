"""Exact OV gauges and weight/data-aware rank truncation on natural rules.

For one bilinear-attention head, O_h V_h is the complete value-to-output map
at fixed attention scores. SVD refactorization is an exact gauge at full
head rank; lower ranks change the function. This does not canonicalize QK,
MLPs, or the complete transformer.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import random

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from train_sports_alias_bracket import draw, evaluate, stack
from train_sports_report_multirule import DeepModel, SPECS


def model_from_checkpoint(path, device):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    cfg = saved['config']
    model = DeepModel(50257, cfg['d_model'], cfg['heads'],
                      SPECS[f"block{cfg['depth']}"], cfg['n_ctx'],
                      norm='rms', residual='lerp',
                      attention='bilinear').to(device)
    model.load_state_dict(saved['state_dict'])
    model.eval()
    return model, cfg


@torch.no_grad()
def activation_grams(model, fit, device):
    attentions = [layer for layer in model.layers if hasattr(layer, 'q1')]
    grams = [torch.zeros((64, 64), dtype=torch.float64)
             for _ in attentions]
    counts = [0]*len(attentions)
    for arm in fit:
        for start in range(0, len(arm['x']), 4):
            rows = {k: v[start:start+4].to(device)
                    for k, v in arm.items()}
            mask = rows['y'] != -100
            hooks = []
            for index, layer in enumerate(attentions):
                def capture(module, inputs, index=index):
                    x = inputs[0][mask].detach().to('cpu', torch.float64)
                    grams[index].add_(x.T @ x)
                    counts[index] += len(x)
                hooks.append(layer.v.register_forward_pre_hook(capture))
            model(rows['x'])
            for hook in hooks:
                hook.remove()
    return [gram/count for gram, count in zip(grams, counts)], counts


def sqrt_gram(gram, ridge_fraction):
    ridge = ridge_fraction*torch.trace(gram)/len(gram)
    value, basis = torch.linalg.eigh(gram+ridge*torch.eye(len(gram)))
    value = value.clamp_min(1e-12)
    sqrt = (basis*value.sqrt()) @ basis.T
    inverse = (basis*value.rsqrt()) @ basis.T
    return sqrt, inverse


def rank_product(matrix, rank, metric):
    if metric is None:
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
        return (u[:, :rank]*s[:rank]) @ vh[:rank]
    sqrt, inverse = metric
    u, s, vh = torch.linalg.svd(matrix @ sqrt, full_matrices=False)
    return ((u[:, :rank]*s[:rank]) @ vh[:rank]) @ inverse


@torch.no_grad()
def replace_ov(model, grams, rank, method, ridge_fraction):
    attentions = [layer for layer in model.layers if hasattr(layer, 'q1')]
    rows = []
    for layer_index, layer in enumerate(attentions):
        d_head = layer.d_head
        metric = (sqrt_gram(grams[layer_index], ridge_fraction)
                  if method == 'data' else None)
        for head in range(layer.n_head):
            sl = slice(head*d_head, (head+1)*d_head)
            o = layer.o.weight[:, sl].detach().to('cpu', torch.float64)
            v = layer.v.weight[sl, :].detach().to('cpu', torch.float64)
            original = o @ v
            reduced = rank_product(original, rank, metric)
            u, s, vh = torch.linalg.svd(reduced, full_matrices=False)
            # Balanced factors fix the internal value-coordinate gauge.
            new_o = torch.zeros_like(o)
            new_v = torch.zeros_like(v)
            new_o[:, :rank] = u[:, :rank]*s[:rank].sqrt()
            new_v[:rank, :] = s[:rank, None].sqrt()*vh[:rank]
            layer.o.weight[:, sl].copy_(new_o.to(layer.o.weight.device,
                                                   layer.o.weight.dtype))
            layer.v.weight[sl, :].copy_(new_v.to(layer.v.weight.device,
                                                 layer.v.weight.dtype))
            error = reduced-original
            weighted = torch.trace(error @ grams[layer_index] @ error.T)
            base = torch.trace(original @ grams[layer_index] @ original.T)
            rows.append({'layer': layer_index, 'head': head,
                         'weight_rel_error': float(
                             torch.linalg.norm(error)/
                             torch.linalg.norm(original)),
                         'input_weighted_rel_error': float(
                             (weighted/base).clamp_min(0).sqrt())})
    return rows


@torch.no_grad()
def max_probe_difference(a, b, panel, device):
    rows = panel[0]
    x = rows['x'][:2].to(device)
    return float((a(x)-b(x)).abs().max())


@torch.no_grad()
def teacher_kl(teacher, candidate, panel, device, batch_size=2):
    """Separate rule-site fidelity from a fixed sparse background probe."""
    sums = {'alias': 0., 'bracket': 0., 'background': 0.}
    counts = {'alias': 0, 'bracket': 0, 'background': 0}
    for arm in panel:
        for start in range(0, len(arm['x']), batch_size):
            row = {k: v[start:start+batch_size].to(device)
                   for k, v in arm.items()}
            base = teacher(row['x']).float()
            test = candidate(row['x']).float()
            ix = torch.arange(len(row['x']), device=device)
            for kind, name in enumerate(('alias', 'bracket')):
                positions = row['site'][:, kind]
                p = F.softmax(base[ix, positions], -1)
                logq = F.log_softmax(test[ix, positions], -1)
                sums[name] += F.kl_div(logq, p,
                                       reduction='none').sum(-1).sum().item()
                counts[name] += len(row['x'])
            positions = torch.arange(0, row['x'].shape[1], 32,
                                     device=device)
            valid = row['y'][:, positions] != -100
            valid &= (positions[None, :] != row['site'][:, :1])
            valid &= (positions[None, :] != row['site'][:, 1:2])
            p = F.softmax(base[:, positions], -1)
            logq = F.log_softmax(test[:, positions], -1)
            divergence = F.kl_div(logq, p, reduction='none').sum(-1)
            sums['background'] += divergence[valid].sum().item()
            counts['background'] += valid.sum().item()
    return {key: sums[key]/counts[key] for key in sums} | \
        {'background_positions': counts['background']}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--fit-worlds', type=int, default=80)
    p.add_argument('--test-worlds', type=int, default=40)
    p.add_argument('--ranks', default='1,2,4,8,16')
    p.add_argument('--ridge-fraction', type=float, default=.001)
    p.add_argument('--seed', type=int, default=41601)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    model, cfg = model_from_checkpoint(args.checkpoint, args.device)
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    rng = random.Random(args.seed)
    fit, fit_attempts = draw(rng, args.fit_worlds, 'train', tokenizer,
                             cfg['n_ctx'], unique=True)
    test, test_attempts = draw(rng, args.test_worlds, 'heldout', tokenizer,
                               cfg['n_ctx'], unique=True)
    assert not {key for key, _ in fit} & {key for key, _ in test}
    fit, test = stack(fit), stack(test)
    grams, counts = activation_grams(model, fit, args.device)
    result = {
        'config': {k: str(v) if isinstance(v, Path) else v
                   for k, v in vars(args).items()},
        'fit_attempts': fit_attempts, 'test_attempts': test_attempts,
        'fit_activation_counts': counts,
        'native': evaluate(model, test, args.device, batch_size=4),
        'rows': [],
        'scope': 'All heads of both attention layers are refactored. QK, '
                 'MLPs, embeddings, and unembedding are unchanged. Full '
                 'head rank must preserve logits. Rank truncation is '
                 'compared at the same rank using weight Frobenius versus '
                 'fit-activation covariance metrics.',
        'caveat': 'Only bracket closure is robustly learned in this model. '
                  'Alias is partial. Empirical sports prose is a natural '
                  'carrier, not a fully sampled next-token DGP. A compact '
                  'OV product is not by itself a legible causal circuit.'}
    for method in ('weight', 'data'):
        for rank in (int(value) for value in args.ranks.split(',')):
            candidate = deepcopy(model)
            errors = replace_ov(candidate, grams, rank, method,
                                args.ridge_fraction)
            probe = max_probe_difference(model, candidate, test,
                                         args.device)
            if rank == 16 and probe > .001:
                raise AssertionError(f'Full-rank gauge changed logits: {probe}')
            score = evaluate(candidate, test, args.device, batch_size=4)
            kl = teacher_kl(model, candidate, test, args.device)
            row = {'method': method, 'rank_per_head': rank,
                   'max_probe_logit_change': probe,
                   'mean_weight_rel_error': sum(
                       item['weight_rel_error'] for item in errors)/len(errors),
                   'mean_input_weighted_rel_error': sum(
                       item['input_weighted_rel_error'] for item in errors)/
                       len(errors),
                   'heldout': score, 'teacher_kl': kl}
            result['rows'].append(row)
            print(method, rank, 'bracket',
                  round(score['bracket']['paired_top1'], 3),
                  'alias', round(score['alias']['paired_top1'], 3),
                  'token NLL', round(score['token_nll'], 3), flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('saved', args.output)


if __name__ == '__main__':
    main()

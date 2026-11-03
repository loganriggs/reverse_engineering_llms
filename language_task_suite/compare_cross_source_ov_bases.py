"""Compare weight and training-activation OV bases on two closure rules.

Basis fitting uses only sports training games and the FineWeb delimiter
training cache. The sports/FineWeb evaluation panels remain untouched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from ablate_cross_source_closure_heads import ROOT
from ablate_cross_source_ov_modes import CRITICAL, factors_topk
from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from eval_sports_ov_canonical_truncation import (
    model_from_checkpoint, rank_product, sqrt_gram)
from train_sports_alias_bracket import draw, evaluate as sports_evaluate, stack
from train_three_source_rule_mixture import joint_evaluate


DELIMITER_CACHE = Path('/home/loganriggs/Coding/neuron_circuits/artifacts/'
                       'corpus_delimiter_pilot/dataset.pt')


@torch.no_grad()
def activation_gram(model, layer, x, y, device, batch_size=4):
    gram = torch.zeros((layer.v.in_features, layer.v.in_features),
                       dtype=torch.float64)
    count = 0
    for start in range(0, len(x), batch_size):
        tokens = x[start:start+batch_size].to(device)
        mask = (y[start:start+batch_size] != -100).to(device)
        def capture(module, inputs):
            nonlocal count
            a = inputs[0][mask].detach().to('cpu', torch.float64)
            gram.add_(a.T @ a)
            count += len(a)
        hook = layer.v.register_forward_pre_hook(capture)
        try:
            model(tokens)
        finally:
            hook.remove()
    return gram / count, count


@torch.no_grad()
def score(model, sports, joint, device):
    s = sports_evaluate(model, sports, device, batch_size=4)
    j = joint_evaluate(model, joint, device, batch_size=2)
    return {'sports_bracket_paired': s['bracket']['candidate_paired'],
            'sports_bracket_top1': s['bracket']['top1'],
            'sports_alias_paired': s['alias']['candidate_paired'],
            'sports_token_nll': s['token_nll'],
            'fineweb_delimiter_paired': j['delimiter_paired_two_way'],
            'fineweb_delimiter_top1': j['delimiter_full_vocab_top1'],
            'fineweb_spelling_paired': j['style_paired_two_way'],
            'fineweb_token_nll': j['token_nll']}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--panel', type=Path, default=ROOT / 'results/'
                   'joint_style_delimiter_fresh_84500_94500.pt')
    p.add_argument('--sports-fit-worlds', type=int, default=128)
    p.add_argument('--delimiter-fit-windows', type=int, default=512)
    p.add_argument('--sports-test-worlds', type=int, default=100)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available()
                   else 'cpu')
    p.add_argument('--output', type=Path, default=ROOT / 'results/'
                   'cross_source_ov_basis_comparison.json')
    a = p.parse_args()
    torch.set_num_threads(4)
    joint = torch.load(a.panel, map_location='cpu', weights_only=True)
    cache = torch.load(DELIMITER_CACHE, map_location='cpu', weights_only=True)
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    fit_records, _ = draw(random.Random(52001), a.sports_fit_worlds,
                          'train', tokenizer, joint['config']['n_ctx'],
                          unique=True)
    fit = stack(fit_records)
    sports_fit_x = torch.cat([arm['x'] for arm in fit])
    sports_fit_y = torch.cat([arm['y'] for arm in fit])
    test_records, _ = draw(random.Random(51943), a.sports_test_worlds,
                           'heldout', tokenizer, joint['config']['n_ctx'],
                           unique=True)
    sports_test = stack(test_records)
    rng = torch.Generator().manual_seed(52002)
    indices = torch.randperm(len(cache['train']['x']),
                             generator=rng)[:a.delimiter_fit_windows]
    deli_fit_x = F.pad(cache['train']['x'][indices],
                       (0, joint['config']['n_ctx']-cache['n_ctx']),
                       value=50256)
    deli_fit_y = F.pad(cache['train']['y'][indices],
                       (0, joint['config']['n_ctx']-cache['n_ctx']),
                       value=-100)
    result = {'fit': {'sports_train_worlds': a.sports_fit_worlds,
                      'delimiter_train_windows': a.delimiter_fit_windows,
                      'sports_train_group': 'train',
                      'fineweb_train_source': 'delimiter training cache'},
              'test': {'sports_heldout_worlds': a.sports_test_worlds,
                       'fineweb_joint_documents': joint['counts']['aligned'],
                       'fineweb_offset': joint['config']['skip']},
              'method': 'Truncate O_h V_h by SVD after whitening the '
                        'input side using the indicated uncentered training '
                        'activation Gram, with ridge 1e-4 mean eigenvalue.',
              'models': {}}
    for seed, head_number in CRITICAL.items():
        path = ROOT / 'results' / (
            f'three_source_rule_mixture_seed{seed}_1000_three_source.pt')
        model, cfg = model_from_checkpoint(path, a.device)
        assert cfg['n_ctx'] == joint['config']['n_ctx']
        layer = [block for block in model.layers if hasattr(block, 'q1')][1]
        width = layer.d_head
        sl = slice(head_number*width, (head_number+1)*width)
        original_o = layer.o.weight[:, sl].detach().clone()
        original_v = layer.v.weight[sl, :].detach().clone()
        product = original_o.double().cpu() @ original_v.double().cpu()
        sports_gram, sports_count = activation_gram(
            model, layer, sports_fit_x, sports_fit_y, a.device)
        deli_gram, deli_count = activation_gram(
            model, layer, deli_fit_x, deli_fit_y, a.device)
        grams = {'weight': None, 'sports_activation': sports_gram,
                 'fineweb_activation': deli_gram,
                 'balanced_activation': (sports_gram+deli_gram)/2}
        baseline = score(model, sports_test, joint, a.device)
        methods = {}
        for name, gram in grams.items():
            metric = None if gram is None else sqrt_gram(gram, 1e-4)
            ranks = []
            for rank in (1, 2, 4, 8, 16):
                reduced = rank_product(product, rank, metric)
                u, s, vh = torch.linalg.svd(reduced, full_matrices=False)
                new_o, new_v = factors_topk(
                    u, s, vh, rank, width, original_o.dtype,
                    original_o.device)
                with torch.no_grad():
                    layer.o.weight[:, sl].copy_(new_o)
                    layer.v.weight[sl, :].copy_(new_v)
                try:
                    value = score(model, sports_test, joint, a.device)
                finally:
                    with torch.no_grad():
                        layer.o.weight[:, sl].copy_(original_o)
                        layer.v.weight[sl, :].copy_(original_v)
                ranks.append({'rank': rank, 'scores': value,
                              'relative_product_frobenius_error': float(
                                  torch.linalg.norm(reduced-product)/
                                  torch.linalg.norm(product))})
                print(seed, name, rank, value, flush=True)
            methods[name] = ranks
        result['models'][str(seed)] = {
            'critical_attention_layer': 1, 'head': head_number,
            'fit_active_tokens': {'sports': sports_count,
                                  'fineweb': deli_count},
            'baseline': baseline, 'methods': methods}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print('saved', a.output, flush=True)


if __name__ == '__main__':
    main()

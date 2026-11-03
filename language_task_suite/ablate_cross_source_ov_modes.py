"""Remove weight-only OV singular modes inside causal closure heads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import torch
from tokenizers import Tokenizer

from ablate_cross_source_closure_heads import ROOT, score
from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from eval_sports_ov_canonical_truncation import model_from_checkpoint
from train_sports_alias_bracket import draw, stack


CRITICAL = {41701: 1, 41702: 3}  # head in second attention layer


def factors_without(u, s, vh, omit, d_head, dtype, device):
    keep = [index for index in range(d_head) if index != omit]
    root = s[keep].sqrt()
    o = torch.zeros((u.shape[0], d_head), dtype=dtype, device=device)
    v = torch.zeros((d_head, vh.shape[1]), dtype=dtype, device=device)
    o[:, :len(keep)] = (u[:, keep] * root).to(dtype=dtype, device=device)
    v[:len(keep), :] = (root[:, None] * vh[keep]).to(dtype=dtype, device=device)
    return o, v


def factors_topk(u, s, vh, rank, d_head, dtype, device):
    o = torch.zeros((u.shape[0], d_head), dtype=dtype, device=device)
    v = torch.zeros((d_head, vh.shape[1]), dtype=dtype, device=device)
    root = s[:rank].sqrt()
    o[:, :rank] = (u[:, :rank] * root).to(dtype=dtype, device=device)
    v[:rank, :] = (root[:, None] * vh[:rank]).to(dtype=dtype, device=device)
    return o, v


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--panel', type=Path, default=ROOT / 'results/'
                   'joint_style_delimiter_fresh_84500_94500.pt')
    p.add_argument('--sports-worlds', type=int, default=100)
    p.add_argument('--sports-seed', type=int, default=51943)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available()
                   else 'cpu')
    p.add_argument('--output', type=Path, default=ROOT / 'results/'
                   'cross_source_closure_ov_mode_ablation.json')
    a = p.parse_args()
    torch.set_num_threads(4)
    joint = torch.load(a.panel, map_location='cpu', weights_only=True)
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    records, _ = draw(random.Random(a.sports_seed), a.sports_worlds,
                      'heldout', tokenizer, joint['config']['n_ctx'],
                      unique=True)
    sports = stack(records)
    output = {'panel': joint['config'], 'sports_worlds': a.sports_worlds,
              'sports_seed': a.sports_seed, 'models': {},
              'method': 'Weight-only SVD of O_h V_h, with one singular '
                        'component removed and the remaining product '
                        'refactored into the original head width.',
              'caveat': 'A singular mode is a weight-canonical direction '
                        'under Euclidean residual geometry, not necessarily '
                        'a human-readable algorithm or data-relevant mode.'}
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
        product = (original_o.double().cpu() @ original_v.double().cpu())
        u, s, vh = torch.linalg.svd(product, full_matrices=False)
        assert (s[width:] < 1e-8*s[0]).all(), 'Head product rank exceeds width'
        baseline = score(model, sports, joint, a.device)
        # The complete weight-only SVD must be an exact OV gauge up to fp32.
        full_o, full_v = factors_without(
            u, s, vh, None, width, original_o.dtype, original_o.device)
        with torch.no_grad():
            layer.o.weight[:, sl].copy_(full_o)
            layer.v.weight[sl, :].copy_(full_v)
        probe = sports[0]['x'][:2].to(a.device)
        refactored = model(probe).detach()
        with torch.no_grad():
            layer.o.weight[:, sl].copy_(original_o)
            layer.v.weight[sl, :].copy_(original_v)
        original_logits = model(probe).detach()
        max_gauge_logit_delta = float((refactored-original_logits).abs().max())
        assert max_gauge_logit_delta < 1e-3, max_gauge_logit_delta
        modes = []
        for mode in range(width):
            replacement_o, replacement_v = factors_without(
                u, s, vh, mode, width, original_o.dtype, original_o.device)
            with torch.no_grad():
                layer.o.weight[:, sl].copy_(replacement_o)
                layer.v.weight[sl, :].copy_(replacement_v)
            try:
                ablated = score(model, sports, joint, a.device)
            finally:
                with torch.no_grad():
                    layer.o.weight[:, sl].copy_(original_o)
                    layer.v.weight[sl, :].copy_(original_v)
            entry = {'mode': mode, 'singular_value': float(s[mode]),
                     'relative_frobenius_energy': float(
                         s[mode]**2 / (s[:width]**2).sum()),
                     'scores': ablated,
                     'drops': {key: baseline[key]-ablated[key]
                               for key in baseline}}
            modes.append(entry)
            print(seed, mode, entry['drops'], flush=True)
        ranks = []
        for rank in (1, 2, 4, 8, 12, 16):
            replacement_o, replacement_v = factors_topk(
                u, s, vh, rank, width, original_o.dtype, original_o.device)
            with torch.no_grad():
                layer.o.weight[:, sl].copy_(replacement_o)
                layer.v.weight[sl, :].copy_(replacement_v)
            try:
                retained = score(model, sports, joint, a.device)
            finally:
                with torch.no_grad():
                    layer.o.weight[:, sl].copy_(original_o)
                    layer.v.weight[sl, :].copy_(original_v)
            ranks.append({'rank': rank, 'energy_retained': float(
                (s[:rank]**2).sum()/(s[:width]**2).sum()),
                'scores': retained})
            print(seed, 'rank', rank, retained, flush=True)
        output['models'][str(seed)] = {
            'attention_layer': 1, 'head': head_number,
            'full_rank_max_probe_logit_delta': max_gauge_logit_delta,
            'baseline': baseline, 'mode_ablations': modes,
            'top_rank_retention': ranks}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(output, indent=2) + '\n')
    print('saved', a.output, flush=True)


if __name__ == '__main__':
    main()

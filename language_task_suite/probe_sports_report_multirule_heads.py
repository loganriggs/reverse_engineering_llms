"""Task-blind weight norm versus two-rule head causality in one trained model."""

import argparse
import hashlib
import json
from pathlib import Path
import random

import torch
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from eval_sports_report_multirule_fresh import evaluate
from train_sports_report_multirule import DeepModel, SPECS, draw, stack


@torch.no_grad()
def weight_scores(model):
    scores = []
    for layer_index, layer in enumerate(model.layers):
        if not hasattr(layer, 'q1'):
            continue
        for head in range(layer.n_head):
            sl = slice(head*layer.d_head, (head+1)*layer.d_head)
            q1, k1 = layer.q1.weight[sl], layer.k1.weight[sl]
            q2, k2 = layer.q2.weight[sl], layer.k2.weight[sl]
            v, o = layer.v.weight[sl], layer.o.weight[:, sl]
            scores.append({'layer': layer_index, 'head': head,
                           'qk1_norm': (q1.T @ k1).norm().item(),
                           'qk2_norm': (q2.T @ k2).norm().item(),
                           'ov_norm': (o @ v).norm().item(),
                           'product_norm': ((q1.T @ k1).norm() *
                                            (q2.T @ k2).norm() *
                                            (o @ v).norm()).item()})
    return scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--n', type=int, default=100)
    p.add_argument('--seed', type=int, default=96901)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    state = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    config = state['config']
    if config['depth'] != 2:
        raise ValueError('Pilot compares the two-block model')
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    records, attempts = draw(random.Random(a.seed), a.n, 'heldout', tokenizer,
                              config['n_ctx'], True,
                              carrier_words=config['carrier_words'])
    batch = stack(records)
    model = DeepModel(50257, config['d_model'], config['heads'],
                      SPECS['block2'], config['n_ctx'],
                      norm='rms', residual='lerp',
                      attention='bilinear').to(a.device)
    model.load_state_dict(state['state_dict'])
    model.eval()
    native = evaluate(model, batch, a.device)
    rows = weight_scores(model)
    for row in rows:
        layer = model.layers[row['layer']]
        head = row['head']
        def zero_head(module, inputs, *, head=head, width=layer.d_head):
            value = inputs[0].clone()
            value[..., head*width:(head+1)*width] = 0
            return (value,)
        hook = layer.o.register_forward_pre_hook(zero_head)
        try:
            altered = evaluate(model, batch, a.device)
        finally:
            hook.remove()
        row['ablated'] = {kind: {
            'paired_top1_drop': native['metrics'][kind]['paired_top1']-
                                altered['metrics'][kind]['paired_top1'],
            'targeted_full_tv_drop':
                native['metrics'][kind]['targeted_full_tv']-
                altered['metrics'][kind]['targeted_full_tv'],
            'edit_direction_drop': native['metrics'][kind]['edit_direction']-
                                   altered['metrics'][kind]['edit_direction'],
        } for kind in ('winning_score', 'player_points', 'margin')}
    result = {'config': {**vars(a), 'checkpoint': str(a.checkpoint),
                         'output': str(a.output), 'draw_attempts': attempts},
              'checkpoint_sha256': hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
              'native': native, 'heads': rows,
              'caveat': 'Task-blind product norm is gauge-sensitive and ignores '
                        'RoPE, input distributions, nonlinear downstream paths, '
                        'and interactions among heads. Ablation is causal but '
                        'does not show a unique or complete circuit. Margin '
                        'behavior fails its eligibility gate.'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    for row in sorted(rows, key=lambda x: -x['product_norm']):
        print(row['layer'], row['head'], round(row['product_norm'], 2),
              {kind: round(row['ablated'][kind]['paired_top1_drop'], 3)
               for kind in ('winning_score', 'player_points')}, flush=True)


if __name__ == '__main__':
    main()

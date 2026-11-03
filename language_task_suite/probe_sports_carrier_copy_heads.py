"""Weight-only OV screening versus causal heads on held-out score copying.

The OV score is deliberately partial: an embedding-to-value-to-output-to-
readout path for token pairs, omitting QK, later nonlinear computation, and
the actual contexts. The second-attention score uses a documented
zero-first-attention tokenwise surrogate. It is not an exact circuit.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from train_sports_report_carrier import DeepModel, SPECS, _stack, draw_pairs, evaluate


@torch.no_grad()
def ov_copy_scores(model, left_targets, right_targets):
    """Average paired score-token response along each head's isolated OV path."""
    table = model.embed.weight
    readout = model.head.weight[left_targets] - model.head.weight[right_targets]
    # With attention zeroed, the first lerp leaves 0.5*embedding for MLP1.
    tables = (table, model.layers[1](0.5 * table))
    result = []
    for layer_index, inputs in ((0, tables[0]), (2, tables[1])):
        layer = model.layers[layer_index]
        source_delta = (F.linear(layer.norm(inputs[left_targets]),
                                 layer.v.weight) -
                        F.linear(layer.norm(inputs[right_targets]),
                                 layer.v.weight))
        for head in range(layer.n_head):
            part = torch.zeros_like(source_delta)
            lo, hi = head * layer.d_head, (head + 1) * layer.d_head
            part[:, lo:hi] = source_delta[:, lo:hi]
            output_delta = F.linear(part, layer.o.weight)
            score = (output_delta * readout).sum(-1)
            result.append({'layer_index': layer_index, 'head': head,
                           'mean_signed_ov_copy_score': score.mean().item(),
                           'mean_absolute_ov_copy_score': score.abs().mean().item(),
                           'positive_fraction': (score > 0).float().mean().item()})
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--pairs', type=int, default=64)
    p.add_argument('--seed', type=int, default=76101)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    state = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    config = state['config']
    if config['query_kind'] != 'winning_score' or state['depth'] != 2:
        raise ValueError('Probe requires the two-block winning-score checkpoint')
    if state['tokenizer_sha256'] != hashlib.sha256(DEFAULT_TOKENIZER.read_bytes()).hexdigest():
        raise ValueError('Tokenizer changed')
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    rows, attempts = draw_pairs(random.Random(a.seed), a.pairs, 'heldout',
                                tokenizer, config['n_ctx'], unique_games=True,
                                carrier_words=config['carrier_words'],
                                query_kind='winning_score')
    pair = tuple(_stack([record[side] for _, record in rows])
                 for side in (0, 1))
    model = DeepModel(50257, config['d_model'], config['heads'],
                      SPECS['block2'], config['n_ctx'],
                      norm='rms', residual='lerp', attention='bilinear').to(a.device)
    model.load_state_dict(state['state_dict'])
    model.eval()
    baseline = evaluate(model, pair, a.device)
    weight_scores = ov_copy_scores(model, pair[0]['target'].to(a.device),
                                   pair[1]['target'].to(a.device))
    head_rows = []
    for layer_index in (0, 2):
        layer = model.layers[layer_index]
        for head in range(layer.n_head):
            def zero_head(module, inputs, *, head=head, width=layer.d_head):
                value = inputs[0].clone()
                value[..., head*width:(head+1)*width] = 0
                return (value,)
            hook = layer.o.register_forward_pre_hook(zero_head)
            try:
                observed = evaluate(model, pair, a.device)
            finally:
                hook.remove()
            head_rows.append({'layer_index': layer_index, 'head': head,
                              'both_variants_two_way': observed['both_variants_two_way'],
                              'drop_both_variants_two_way':
                                  baseline['both_variants_two_way'] -
                                  observed['both_variants_two_way'],
                              'correct_shift_fraction':
                                  observed['correct_shift_fraction'],
                              'drop_mean_correct_shift_logits':
                                  baseline['mean_correct_shift_logits'] -
                                  observed['mean_correct_shift_logits'],
                              'site_nll_increase':
                                  observed['site_nll']-baseline['site_nll']})
    result = {'config': {'checkpoint': str(a.checkpoint), 'pairs': a.pairs,
                         'seed': a.seed, 'attempts': attempts,
                         'carrier_words': config['carrier_words']},
              'checkpoint_sha256': hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
              'baseline': baseline, 'weight_only_ov_scores': weight_scores,
              'head_ablations': head_rows,
              'caveat': 'OV score uses only weights and the selected answer-token '
                        'pairs. It omits QK, context, and downstream nonlinear paths; '
                        'the second-layer input is a zero-first-attention surrogate. '
                        'Causal ablations use full held-out documents. No unique '
                        'internal circuit is inferred from ranking agreement.'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print('baseline', baseline, flush=True)
    print('heads', [(r['layer_index'], r['head'],
                     round(r['drop_both_variants_two_way'], 3),
                     round(r['site_nll_increase'], 3))
                    for r in head_rows], flush=True)


if __name__ == '__main__':
    main()

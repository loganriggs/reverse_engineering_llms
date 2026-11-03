"""Coarse causal screen for shared sports/FineWeb closure heads.

This only removes complete attention heads. It does not identify a QK/OV path
or prove that a head contains one reusable computation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import torch
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from eval_sports_ov_canonical_truncation import model_from_checkpoint
from train_sports_alias_bracket import draw, evaluate as sports_evaluate, stack
from train_three_source_rule_mixture import joint_evaluate


ROOT = Path(__file__).resolve().parent


def score(model, sports, joint, device):
    s = sports_evaluate(model, sports, device, batch_size=4)
    j = joint_evaluate(model, joint, device, batch_size=2)
    return {'sports_bracket_paired': s['bracket']['candidate_paired'],
            'sports_bracket_top1': s['bracket']['top1'],
            'sports_alias_paired': s['alias']['candidate_paired'],
            'fineweb_delimiter_paired': j['delimiter_paired_two_way'],
            'fineweb_delimiter_top1': j['delimiter_full_vocab_top1'],
            'fineweb_spelling_paired': j['style_paired_two_way']}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--panel', type=Path, default=ROOT / 'results/'
                   'joint_style_delimiter_fresh_84500_94500.pt')
    p.add_argument('--sports-worlds', type=int, default=100)
    p.add_argument('--sports-seed', type=int, default=51943)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available()
                   else 'cpu')
    p.add_argument('--output', type=Path, default=ROOT / 'results/'
                   'cross_source_closure_head_ablation.json')
    a = p.parse_args()
    torch.set_num_threads(4)
    joint = torch.load(a.panel, map_location='cpu', weights_only=True)
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    records, attempts = draw(random.Random(a.sports_seed), a.sports_worlds,
                             'heldout', tokenizer, joint['config']['n_ctx'],
                             unique=True)
    sports = stack(records)
    results = {'panel': joint['config'], 'sports_worlds': a.sports_worlds,
               'sports_attempts': attempts, 'device': a.device,
               'models': {},
               'caveat': 'Complete-head output ablations are a coarse necessity '
                         'screen. Shared score drops can reflect separate '
                         'features in one head, nonlinear interactions, or '
                         'general disruption; they do not identify a common '
                         'QK/OV path.'}
    for seed in (41701, 41702):
        path = ROOT / 'results' / (
            f'three_source_rule_mixture_seed{seed}_1000_three_source.pt')
        model, cfg = model_from_checkpoint(path, a.device)
        assert cfg['n_ctx'] == joint['config']['n_ctx']
        baseline = score(model, sports, joint, a.device)
        heads = []
        for layer_number, layer in enumerate(model.layers):
            if not hasattr(layer, 'q1'):
                continue
            for head_number in range(layer.n_head):
                sl = slice(head_number*layer.d_head,
                           (head_number+1)*layer.d_head)
                with torch.no_grad():
                    original = layer.o.weight[:, sl].clone()
                    layer.o.weight[:, sl].zero_()
                try:
                    ablated = score(model, sports, joint, a.device)
                finally:
                    with torch.no_grad():
                        layer.o.weight[:, sl].copy_(original)
                heads.append({'layer': layer_number, 'head': head_number,
                              'scores': ablated,
                              'drops': {key: baseline[key]-ablated[key]
                                        for key in baseline}})
                print(seed, layer_number, head_number,
                      heads[-1]['drops'], flush=True)
        results['models'][str(seed)] = {'baseline': baseline,
                                        'head_ablations': heads}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(results, indent=2) + '\n')
    print('saved', a.output, flush=True)


if __name__ == '__main__':
    main()

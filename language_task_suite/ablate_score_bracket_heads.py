"""Singleton head-output ablations on the heldout score x bracket panel."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

import torch
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from eval_score_bracket_ov import teacher_kl
from eval_sports_ov_canonical_truncation import activation_grams, model_from_checkpoint
from train_sports_score_bracket import draw, evaluate, stack
from train_sports_score_bracket_mixture import separate_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--fit-plans-per-rule', type=int, default=40)
    parser.add_argument('--test-worlds', type=int, default=28)
    parser.add_argument('--carrier-words', type=int, default=100)
    parser.add_argument('--seed', type=int, default=55901)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    teacher, cfg = model_from_checkpoint(args.checkpoint, args.device)
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    rng = random.Random(args.seed)
    score_rows, score_games, _ = separate_data(
        rng, args.fit_plans_per_rule, 'score', tokenizer,
        cfg['n_ctx'], args.carrier_words)
    bracket_rows, bracket_games, _ = separate_data(
        rng, args.fit_plans_per_rule, 'bracket', tokenizer,
        cfg['n_ctx'], args.carrier_words)
    rows = score_rows + bracket_rows
    fit = ({key: torch.stack([row[key] for row in rows])
            for key in ('x', 'y', 'site')},)
    records, attempts = draw(
        random.Random(args.seed), args.test_worlds, 'heldout',
        tokenizer, cfg['n_ctx'], args.carrier_words, 'winning_score',
        unique=True)
    panel = stack(records)
    grams, counts = activation_grams(teacher, fit, args.device)
    native = evaluate(teacher, panel, args.device)
    report = {
        'config': {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items()},
        'fit_games': {'score': score_games, 'bracket': bracket_games},
        'fit_activation_counts': counts,
        'test_attempts': attempts,
        'test_game_sha256': [hashlib.sha256(repr(key).encode()).hexdigest()
                             for key, _ in records],
        'native': native, 'heads': [],
        'scope': 'Zero each head output column in O, with other parameters '
                 'frozen. OV norms use separate-rule train activation '
                 'covariance; causal fidelity uses joint heldout four-arm '
                 'reports. Singleton ablations do not isolate full circuits.'}
    attentions = [layer for layer in teacher.layers if hasattr(layer, 'q1')]
    for layer_index, layer in enumerate(attentions):
        for head in range(layer.n_head):
            sl = slice(head*layer.d_head, (head+1)*layer.d_head)
            o = layer.o.weight[:, sl].detach().cpu().double()
            v = layer.v.weight[sl, :].detach().cpu().double()
            ov = o @ v
            weighted = torch.trace(ov @ grams[layer_index] @ ov.T)
            student = deepcopy(teacher)
            student_layer = [x for x in student.layers if hasattr(x, 'q1')][layer_index]
            with torch.no_grad():
                student_layer.o.weight[:, sl].zero_()
            heldout = evaluate(student, panel, args.device)
            divergence = teacher_kl(teacher, student, panel, args.device)
            item = {
                'layer': layer_index, 'head': head,
                'weight_ov_frobenius': float(torch.linalg.norm(ov)),
                'activation_weighted_ov_norm': float(weighted.clamp_min(0).sqrt()),
                'heldout': heldout, 'teacher_kl': divergence,
                'paired_bracket_loss': native['bracket']['paired_top1'] -
                                       heldout['bracket']['paired_top1'],
                'paired_score_loss': native['numeric']['paired_top1'] -
                                     heldout['numeric']['paired_top1']}
            report['heads'].append(item)
            print(layer_index, head, 'bracket',
                  round(heldout['bracket']['paired_top1'], 3), 'score',
                  round(heldout['numeric']['paired_top1'], 3), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print('saved', args.output)


if __name__ == '__main__':
    main()

"""Separate exact BPE-token errors from the intended closing-bracket style."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from train_corpus_delimiter_bilinear import _model


def classify(token, correct):
    other = ']' if correct == ')' else ')'
    if token == correct:
        return 'exact_character_token'
    if token.endswith(correct) and other not in token:
        return 'right_style_ends_token'
    if correct in token and other not in token:
        return 'right_style_inside_token'
    if other in token and correct not in token:
        return 'wrong_style_inside_token'
    if other in token and correct in token:
        return 'both_styles_inside_token'
    return 'no_closing_bracket_in_token'


@torch.no_grad()
def audit(model, panel, tokenizer, device, batch_size=8):
    counts = Counter()
    wrong = Counter()
    targets = Counter()
    n = len(panel[(0, 'round')]['x'])
    for (shift, style), row in panel.items():
        correct = ')' if style == 'round' else ']'
        for start in range(0, n, batch_size):
            x = row['x'][start:start + batch_size].to(device)
            site = row['site'][start:start + batch_size, 0].to(device)
            target = row['target'][start:start + batch_size, 0]
            ix = torch.arange(len(x), device=device)
            predicted = model(x).float()[ix, site].argmax(-1).cpu()
            for p, t in zip(predicted.tolist(), target.tolist(), strict=True):
                token = tokenizer.id_to_token(p)
                expected = tokenizer.id_to_token(t)
                assert expected.endswith(correct), (expected, correct)
                counts['total'] += 1
                counts['exact_top1'] += p == t
                counts[classify(token, correct)] += 1
                if p != t:
                    wrong[token] += 1
                    targets[expected] += 1
    right_style = sum(counts[key] for key in (
        'exact_character_token', 'right_style_ends_token',
        'right_style_inside_token'))
    rates = {key: value / counts['total'] for key, value in counts.items()
             if key != 'total'}
    rates['right_style_anywhere'] = right_style / counts['total']
    return {'n_documents': n, 'n_arm_sites': counts['total'],
            'counts': dict(counts), 'rates': rates,
            'top_wrong_tokens': wrong.most_common(12),
            'top_missed_target_tokens': targets.most_common(12)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    p.add_argument('--joint-cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    torch.set_num_threads(4)
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    panel = torch.load(a.joint_cache, map_location='cpu', weights_only=True)
    report = {'panel': str(a.joint_cache), 'models': {},
              'scope': 'Prediction at bracket-closer BPE site on four edited '
                       'arms. Token contains/ends-with style is a coarse '
                       'character-rule diagnostic; it does not establish '
                       'exact token prediction or grammatical continuation.'}
    for path in a.checkpoints:
        saved = torch.load(path, map_location='cpu', weights_only=True)
        model = _model(50257, 256, 2, 64, 4, a.device, 'none')
        model.load_state_dict(saved['state_dict'])
        model.eval()
        report['models'][path.stem] = audit(model, panel, tokenizer, a.device)
        print(path.stem, report['models'][path.stem]['rates'], flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print('saved', a.output, flush=True)


if __name__ == '__main__':
    main()

"""Compare site and edit-effect supervision for a rank-limited OV reduction.

The fit panel uses only training-range FineWeb source previews. Evaluation
previews are source-disjoint and never enter the optimizer. No raw text or
token sequences are written to the aggregate result.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import random

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from eval_corpus_rule_reference_ov_rank import fidelity
from eval_corpus_rule_reference_transfer import one_kind, pair_for_text
from fit_corpus_list_bracket_ov import trainable_ov, unused_channel_max
from ov_linear_reduction import activation_grams, replace_ov
from train_corpus_delimiter_bilinear import _model


def pairs_from_previews(path, tokenizer, limit, seed):
    rows = json.loads(path.read_text())
    found, reasons = [], Counter()
    for source in rows:
        pair, reason = pair_for_text(source['preview'], tokenizer,
                                     source['index'], 'bracket', 256, 2048)
        reasons[reason] += 1
        if pair is not None:
            found.append((source['sha256'], pair))
    if len(found) < limit:
        raise RuntimeError(f'Only {len(found)} eligible pairs in {path}')
    rng = random.Random(seed)
    chosen = rng.sample(found, limit)
    return [pair for _, pair in chosen], {sha for sha, _ in chosen}, dict(reasons), rows


def paired_tensors(records, device):
    n = len(records)
    return {
        'x': torch.cat((torch.stack([r['x_natural'] for r in records]),
                        torch.stack([r['x_edit'] for r in records]))).to(device),
        'y': torch.cat((torch.stack([r['y_natural'] for r in records]),
                        torch.stack([r['y_edit'] for r in records]))).to(device),
        'pn': torch.tensor([r['pos_natural'] for r in records], device=device),
        'pe': torch.tensor([r['pos_edit'] for r in records], device=device),
        'tn': torch.tensor([r['token_natural'] for r in records], device=device),
        'te': torch.tensor([r['token_edit'] for r in records], device=device),
        'n': n,
    }


def loss_on_pairs(teacher, student, records, mode, background_weight, device):
    batch = paired_tensors(records, device)
    n = batch['n']
    ix = torch.arange(n, device=device)
    with torch.no_grad():
        reference = teacher(batch['x']).float()
    output = student(batch['x']).float()
    rt_n, rt_e = reference[:n][ix, batch['pn']], reference[n:][ix, batch['pe']]
    st_n, st_e = output[:n][ix, batch['pn']], output[n:][ix, batch['pe']]
    site_kl = F.kl_div(torch.cat((st_n, st_e)).log_softmax(-1),
                       torch.cat((rt_n, rt_e)).softmax(-1),
                       reduction='batchmean')
    margin = lambda a, b: a[ix, batch['tn']] - a[ix, batch['te']] - (
        b[ix, batch['tn']] - b[ix, batch['te']])
    effect_t = margin(rt_n, rt_e)
    effect_s = margin(st_n, st_e)
    effect_mse = (effect_s - effect_t).square().mean()
    # Same background sample in both modes, excluding each edited site.
    positions = torch.arange(0, batch['x'].shape[1], 32, device=device)
    sites = torch.cat((batch['pn'], batch['pe']))
    valid = ((batch['y'][:, positions] != -100) &
             (positions[None, :] != sites[:, None]))
    background_kl = F.kl_div(output[:, positions][valid].log_softmax(-1),
                             reference[:, positions][valid].softmax(-1),
                             reduction='batchmean')
    loss = ((site_kl if mode == 'site' else .1 * site_kl + effect_mse) +
            background_weight * background_kl)
    return loss, {'site_kl': float(site_kl.detach()),
                  'effect_mse': float(effect_mse.detach()),
                  'background_kl': float(background_kl.detach())}


def score(teacher, student, records, device, tokenizer):
    with torch.no_grad():
        return {'behavior': one_kind(student, records, device, 8, tokenizer),
                'fidelity': fidelity(teacher, student, records, device, 4)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--train-preview', type=Path, required=True)
    p.add_argument('--test-preview', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--fit-pairs', type=int, default=128)
    p.add_argument('--test-pairs', type=int, default=200)
    p.add_argument('--rank', type=int, default=4)
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--batch-pairs', type=int, default=2)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--background-weight', type=float, default=.03)
    p.add_argument('--seed', type=int, default=46482)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available()
                   else 'cpu')
    a = p.parse_args()
    torch.set_num_threads(4)
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    fit, fit_hashes, fit_reasons, train_rows = pairs_from_previews(
        a.train_preview, tokenizer, a.fit_pairs, a.seed)
    test, test_hashes, test_reasons, test_rows = pairs_from_previews(
        a.test_preview, tokenizer, a.test_pairs, a.seed + 1)
    if fit_hashes & test_hashes:
        raise AssertionError('Training and test source hashes overlap')
    if {r['offset'] for r in fit} & {r['offset'] for r in test}:
        raise AssertionError('Training and test source offsets overlap')
    state = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    if Path(state['identity'][0]) != a.cache.resolve():
        raise ValueError('Checkpoint does not match training cache')
    _, depth, width, heads = state['identity'][:4]
    teacher = _model(50257, 256, depth, width, heads, a.device, 'rms')
    teacher.load_state_dict(state['model'])
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    cache = torch.load(a.cache, map_location='cpu', weights_only=True)
    gram_rows = {key: value[:128] for key, value in
                 cache['train']['background'].items() if key in ('x', 'y')}
    grams, gram_counts = activation_grams(teacher, gram_rows, a.device)
    report = {
        'config': {k: str(v) if isinstance(v, Path) else v
                   for k, v in vars(a).items()},
        'checkpoint_step': state['step'],
        'fit_source_range': [train_rows[0]['index'], train_rows[-1]['index']],
        'test_source_range': [test_rows[0]['index'], test_rows[-1]['index']],
        'fit_reasons': fit_reasons, 'test_reasons': test_reasons,
        'fit_pairs': len(fit), 'test_pairs': len(test),
        'fit_test_hash_overlap': 0, 'activation_counts': gram_counts,
        'teacher': score(teacher, teacher, test, a.device, tokenizer),
        'methods': {},
        'scope': 'Rank-limited per-head OV factors only; QK, MLP, embedding '
                 'and unembedding frozen. All fitting uses train-range '
                 'source pairs; heldout source pairs are evaluation only. '
                 'Edit-effect loss matches the teacher change in the '
                 'natural-vs-square bracket logit margin. This is targeted '
                 'causal fidelity, not independent task discovery or a '
                 'unique canonical circuit.',
    }
    for method in ('weight', 'data', 'site', 'effect'):
        student = deepcopy(teacher).eval()
        reduction = replace_ov(student, grams, a.rank,
                               'data' if method == 'data' else 'weight')
        row = {'reduction_errors': reduction}
        if method in ('site', 'effect'):
            trainable = trainable_ov(student, a.rank)
            optimizer = torch.optim.Adam(trainable, lr=a.lr)
            rng = random.Random(a.seed)
            losses = []
            for step in range(a.steps):
                batch = rng.sample(fit, a.batch_pairs)
                loss, terms = loss_on_pairs(
                    teacher, student, batch, method, a.background_weight,
                    a.device)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError(f'Nonfinite fit loss {method} {step}')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append({'loss': float(loss.detach()), **terms})
                if (step + 1) % 100 == 0:
                    print(method, step + 1, round(losses[-1]['loss'], 4),
                          flush=True)
            row['last50_fit'] = {
                key: sum(item[key] for item in losses[-50:]) /
                     len(losses[-50:]) for key in losses[-1]}
            row['unused_channel_max_abs'] = unused_channel_max(
                student, a.rank)
            assert row['unused_channel_max_abs'] == 0.
        row['test'] = score(teacher, student, test, a.device, tokenizer)
        report['methods'][method] = row
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(report, indent=2) + '\n')
        f = row['test']['fidelity']
        print(method, 'effect MAE', round(f['effect_mean_abs_error'], 4),
              'site KL', round(f['site_teacher_kl'], 4),
              'bg KL', round(f['sampled_background_teacher_kl'], 4),
              flush=True)
    print('saved', a.output, flush=True)


if __name__ == '__main__':
    main()

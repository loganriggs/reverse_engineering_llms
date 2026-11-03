"""Train bilinear LMs on a natural-report carrier with one controlled rule.

The human source prose and reconstructed token cache stay local. Only
aggregate metrics are written to the requested JSON output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from sports_report_carrier_dgp import (
    increment_winning_score, realize, sample_plan,
)
import sports_report_carrier_dgp as carrier_dgp


OLD_REPO = Path(__file__).resolve().parents[2]/'reverse_engineer/tensor_language'
if not (OLD_REPO/'deep_model.py').is_file():
    raise FileNotFoundError(OLD_REPO/'deep_model.py')
sys.modules.pop('model', None)
sys.path.insert(0, str(OLD_REPO))
from deep_model import DeepModel, SPECS  # noqa: E402


PAD = 50256


def aligned(doc, tokenizer, n_ctx):
    encoded = tokenizer.encode(doc.text, add_special_tokens=False)
    indices = [i for i, (start, end) in enumerate(encoded.offsets)
               if start <= doc.site.start < end]
    if len(indices) != 1:
        return None
    target = indices[0]
    if target < 1 or encoded.offsets[target][1] != doc.site.end or \
            len(encoded.ids)-1 > n_ctx or target >= len(encoded.ids)-1:
        return None
    x = torch.full((n_ctx,), PAD, dtype=torch.long)
    y = torch.full((n_ctx,), -100, dtype=torch.long)
    x[:len(encoded.ids)-1] = torch.tensor(encoded.ids[:-1])
    y[:len(encoded.ids)-1] = torch.tensor(encoded.ids[1:])
    assert y[target-1].item() == encoded.ids[target]
    return {'x': x, 'y': y, 'site': target-1,
            'target': encoded.ids[target]}


def pair(plan, tokenizer, n_ctx):
    original = aligned(realize(plan), tokenizer, n_ctx)
    changed = aligned(realize(increment_winning_score(plan)),
                      tokenizer, n_ctx)
    if original is None or changed is None or \
            original['target'] == changed['target']:
        return None
    return original, changed


def _stack(rows):
    return {key: torch.stack([row[key] for row in rows])
            if key in ('x', 'y') else
            torch.tensor([row[key] for row in rows], dtype=torch.long)
            for key in ('x', 'y', 'site', 'target')}


def draw_pairs(rng, count, source_group, tokenizer, n_ctx,
               unique_games=False, excluded_keys=(),
               carrier_words=100, query_kind='margin'):
    rows = []
    seen = set(excluded_keys)
    attempts = 0
    while len(rows) < count:
        attempts += 1
        if attempts > 100*count:
            raise RuntimeError('Too few aligned, distinct carrier pairs')
        plan = sample_plan(rng, source='rotowire', group=source_group,
                           carrier_words=carrier_words,
                           query_kind=query_kind)
        if unique_games and plan.source_game_key in seen:
            continue
        record = pair(plan, tokenizer, n_ctx)
        if record is None:
            continue
        rows.append((plan.source_game_key, record))
        seen.add(plan.source_game_key)
    return rows, attempts


@torch.no_grad()
def evaluate(model, pairs, device, batch_size=8):
    model.eval()
    totals = dict(n=0, token_nll=0., tokens=0, site_nll=0.,
                  top1=0, two_way=0, both_two_way=0,
                  correct_shift=0, shift_sum=0.)
    left, right = pairs
    for start in range(0, len(left['x']), batch_size):
        end = min(start+batch_size, len(left['x']))
        rows = [{k: v[start:end].to(device) for k, v in side.items()}
                for side in (left, right)]
        logits = [model(row['x']).float() for row in rows]
        batch_n = end-start
        ix = torch.arange(batch_n, device=device)
        predictions = []
        for row, scores in zip(rows, logits):
            totals['token_nll'] += F.cross_entropy(
                scores.flatten(0, 1), row['y'].flatten(),
                ignore_index=-100, reduction='sum').item()
            totals['tokens'] += (row['y'] != -100).sum().item()
            site_scores = scores[ix, row['site']]
            logp = site_scores.log_softmax(-1)
            totals['site_nll'] -= logp[ix, row['target']].sum().item()
            totals['top1'] += (site_scores.argmax(-1) == row['target']).sum().item()
            predictions.append(site_scores)
        left_target, right_target = rows[0]['target'], rows[1]['target']
        left_wins = predictions[0][ix, left_target] > \
            predictions[0][ix, right_target]
        right_wins = predictions[1][ix, right_target] > \
            predictions[1][ix, left_target]
        totals['two_way'] += left_wins.sum().item()+right_wins.sum().item()
        totals['both_two_way'] += (left_wins & right_wins).sum().item()
        shift = ((predictions[1][ix, right_target]-
                  predictions[1][ix, left_target])-
                 (predictions[0][ix, right_target]-
                  predictions[0][ix, left_target]))
        totals['correct_shift'] += (shift > 0).sum().item()
        totals['shift_sum'] += shift.sum().item()
        totals['n'] += batch_n
    n = totals['n']
    return {'pairs': n, 'token_nll': totals['token_nll']/totals['tokens'],
            'site_nll': totals['site_nll']/(2*n),
            'full_vocab_top1': totals['top1']/(2*n),
            'two_way_accuracy': totals['two_way']/(2*n),
            'both_variants_two_way': totals['both_two_way']/n,
            'correct_shift_fraction': totals['correct_shift']/n,
            'mean_correct_shift_logits': totals['shift_sum']/n}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--depths', default='1,2,4')
    p.add_argument('--train-pairs', type=int, default=2000)
    p.add_argument('--val-pairs', type=int, default=32)
    p.add_argument('--test-pairs', type=int, default=64)
    p.add_argument('--steps', type=int, default=500)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--eval-batch-size', type=int, default=8)
    p.add_argument('--n-ctx', type=int, default=320)
    p.add_argument('--carrier-words', type=int, default=100)
    p.add_argument('--query-kind', choices=('margin', 'winning_score'),
                   default='margin')
    p.add_argument('--d-model', type=int, default=64)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--site-weight', type=float, default=50.)
    p.add_argument('--lr', type=float, default=.0005)
    p.add_argument('--seed', type=int, default=93401)
    p.add_argument('--save-every', type=int, default=100)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available()
                   else 'cpu')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint-dir', type=Path, default=Path(
        '/home/loganriggs/Coding/neuron_circuits/artifacts/'
        'sports_report_carrier/checkpoints'))
    a = p.parse_args()
    if a.site_weight < 1 or a.batch_size > 2*a.train_pairs or \
            a.save_every < 1:
        raise ValueError('Invalid site weight or batch size')
    torch.set_num_threads(4)
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    rng = random.Random(a.seed)
    train, train_attempts = draw_pairs(rng, a.train_pairs, 'train',
                                       tokenizer, a.n_ctx,
                                       carrier_words=a.carrier_words,
                                       query_kind=a.query_kind)
    val, val_attempts = draw_pairs(rng, a.val_pairs, 'heldout',
                                   tokenizer, a.n_ctx, True,
                                   carrier_words=a.carrier_words,
                                   query_kind=a.query_kind)
    test, test_attempts = draw_pairs(
        rng, a.test_pairs, 'heldout', tokenizer, a.n_ctx, True,
        {key for key, _ in val}, a.carrier_words, a.query_kind)
    assert not {key for key, _ in train} & \
        {key for key, _ in val+test}
    def stacked(rows):
        return tuple(_stack([record[side] for _, record in rows])
                     for side in (0, 1))
    training, validation, testing = map(stacked, (train, val, test))
    xtrain = torch.cat([training[0]['x'], training[1]['x']])
    ytrain = torch.cat([training[0]['y'], training[1]['y']])
    strain = torch.cat([training[0]['site'], training[1]['site']])
    output = {'config': {**vars(a), 'output': str(a.output),
                         'checkpoint_dir': str(a.checkpoint_dir)},
              'tokenizer_sha256': hashlib.sha256(
                  DEFAULT_TOKENIZER.read_bytes()).hexdigest(),
              'carrier_dgp_sha256': hashlib.sha256(
                  Path(carrier_dgp.__file__).read_bytes()).hexdigest(),
              'data': {'train_pairs': len(train), 'val_pairs': len(val),
                       'test_pairs': len(test),
                       'train_unique_games': len({key for key, _ in train}),
                       'train_attempts': train_attempts,
                       'val_attempts': val_attempts,
                       'test_attempts': test_attempts},
              'objective': 'full next-token CE plus specified site weight; '
                           f'paired original/+1 winning score, '
                           f'{a.query_kind} target on empirical '
                           'RotoWire carrier',
              'caveat': 'Only the inserted score and final margin have '
                        'explicit rules; source prose has no symbolic '
                        'oracle. Numeric and first-result screens are '
                        'conservative but not full semantic certification.',
              'results': {}}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for depth in [int(value) for value in a.depths.split(',')]:
        if depth not in (1, 2, 3, 4):
            raise ValueError(depth)
        torch.manual_seed(a.seed+depth)
        model = DeepModel(50257, a.d_model, a.heads,
                          SPECS[f'block{depth}'], a.n_ctx,
                          norm='rms', residual='lerp',
                          attention='bilinear').to(a.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr)
        initial = evaluate(model, validation, a.device, a.eval_batch_size)
        fingerprint = hashlib.sha256(json.dumps({
            'query_kind': a.query_kind, 'carrier_words': a.carrier_words,
            'depth': depth, 'd_model': a.d_model, 'heads': a.heads,
            'n_ctx': a.n_ctx, 'site_weight': a.site_weight, 'lr': a.lr,
            'seed': a.seed, 'train_pairs': a.train_pairs,
            'val_pairs': a.val_pairs, 'test_pairs': a.test_pairs,
            'batch_size': a.batch_size,
            'carrier_dgp_sha256': output['carrier_dgp_sha256'],
            'tokenizer_sha256': output['tokenizer_sha256']},
            sort_keys=True).encode()).hexdigest()
        resume_path = a.checkpoint_dir / (
            f'resume_{a.query_kind}_words{a.carrier_words}_depth{depth}'
            f'_site{a.site_weight:g}_seed{a.seed}.pt')
        losses = []
        completed = 0
        if a.resume and resume_path.exists():
            state = torch.load(resume_path, map_location=a.device,
                               weights_only=True)
            if state['fingerprint'] != fingerprint:
                raise ValueError('Resume checkpoint configuration changed')
            model.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer'])
            completed = state['completed']
            losses = state['losses']
            torch.set_rng_state(state['torch_rng'].cpu())
            if a.device.startswith('cuda'):
                torch.cuda.set_rng_state_all(state['cuda_rng'])
            if completed > a.steps:
                raise ValueError('Checkpoint is beyond requested steps')
            print('resumed', depth, completed, flush=True)

        def save_resume():
            state = {'fingerprint': fingerprint,
                     'model': {key: value.detach().cpu()
                               for key, value in model.state_dict().items()},
                     'optimizer': optimizer.state_dict(),
                     'completed': completed, 'losses': losses,
                     'torch_rng': torch.get_rng_state(),
                     'cuda_rng': torch.cuda.get_rng_state_all()
                     if a.device.startswith('cuda') else []}
            temporary = resume_path.with_suffix('.tmp')
            torch.save(state, temporary)
            os.replace(temporary, resume_path)
        clock = time.time()
        try:
            for step in range(completed, a.steps):
                model.train()
                indices = torch.randint(len(xtrain), (a.batch_size,))
                x = xtrain[indices].to(a.device)
                y = ytrain[indices].to(a.device)
                site = strain[indices].to(a.device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x).float()
                per = F.cross_entropy(logits.flatten(0, 1), y.flatten(),
                                      ignore_index=-100,
                                      reduction='none').view_as(y)
                weights = (y != -100).float()
                weights[torch.arange(len(x), device=a.device), site] = \
                    a.site_weight
                loss = (per*weights).sum()/weights.sum()
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f'Nonfinite loss at depth{depth} step{step}')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                losses.append(loss.item())
                completed = step+1
                if completed % a.save_every == 0:
                    save_resume()
                if completed % 100 == 0:
                    print(depth, completed, round(loss.item(), 4),
                          flush=True)
        except KeyboardInterrupt:
            save_resume()
            raise
        save_resume()
        result = {'parameters': model.n_params,
                  'initial_validation': initial,
                  'train_loss_last50': sum(losses[-50:])/len(losses[-50:]),
                  'validation': evaluate(model, validation, a.device,
                                         a.eval_batch_size),
                  'test': evaluate(model, testing, a.device,
                                   a.eval_batch_size),
                  'elapsed_seconds': round(time.time()-clock, 2)}
        output['results'][str(depth)] = result
        a.output.write_text(json.dumps(output, indent=2)+'\n')
        checkpoint = a.checkpoint_dir / (
            f'{a.query_kind}_words{a.carrier_words}_depth{depth}'
            f'_site{a.site_weight:g}_seed{a.seed}.pt')
        torch.save({'state_dict': {key: value.detach().cpu()
                                   for key, value in model.state_dict().items()},
                    'config': output['config'], 'depth': depth,
                    'tokenizer_sha256': output['tokenizer_sha256']}, checkpoint)
    print('saved', a.output, flush=True)


if __name__ == '__main__':
    main()

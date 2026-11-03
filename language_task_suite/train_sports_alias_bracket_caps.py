"""Bilinear LM pilot on three simultaneous natural-report rule sites."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from sports_alias_bracket_caps_dgp import realize, sample_plan
from train_sports_report_multirule import DeepModel, PAD, SPECS


KINDS = ('alias', 'bracket', 'capitalization')
ARMS = tuple((a, b, c) for a in ('a', 'b')
             for b in ('round', 'square')
             for c in ('period', 'semicolon'))
ARM_INDEX = {arm: i for i, arm in enumerate(ARMS)}


def aligned(doc, tokenizer, n_ctx):
    enc = tokenizer.encode(doc.text, add_special_tokens=False)
    sites, targets = [], []
    for site in doc.sites:
        found = [i for i, (start, end) in enumerate(enc.offsets)
                 if start <= site.start < end]
        if (len(found) != 1 or found[0] < 1 or
                found[0] >= n_ctx or
                enc.offsets[found[0]][1] != site.end):
            return None
        sites.append(found[0]-1)
        targets.append(enc.ids[found[0]])
    length = min(len(enc.ids)-1, n_ctx, max(sites)+41)
    x = torch.full((n_ctx,), PAD, dtype=torch.long)
    y = torch.full((n_ctx,), -100, dtype=torch.long)
    x[:length] = torch.tensor(enc.ids[:length])
    y[:length] = torch.tensor(enc.ids[1:length+1])
    return {'x': x, 'y': y,
            'site': torch.tensor(sites, dtype=torch.long),
            'target': torch.tensor(targets, dtype=torch.long)}


def octet(plan, tokenizer, n_ctx):
    rows = tuple(aligned(realize(plan, *arm), tokenizer, n_ctx)
                 for arm in ARMS)
    if any(row is None for row in rows):
        return None
    for i, left in enumerate(ARMS):
        for j, right in enumerate(ARMS):
            if i >= j:
                continue
            for kind in range(3):
                assert (rows[i]['target'][kind] != rows[j]['target'][kind]) \
                    == (left[kind] != right[kind])
    return rows


def draw(rng, count, group, tokenizer, n_ctx, unique=False, excluded=()):
    out, seen, attempts = [], set(excluded), 0
    while len(out) < count:
        attempts += 1
        if attempts > count*500:
            raise RuntimeError('Too few aligned three-rule reports')
        plan = sample_plan(rng, group)
        key = plan.pair.alias.game_key
        if unique and key in seen:
            continue
        rows = octet(plan, tokenizer, n_ctx)
        if rows is None:
            continue
        out.append((key, rows))
        seen.add(key)
    return out, attempts


def stack(records):
    return tuple({key: torch.stack([rows[i][key] for _, rows in records])
                  for key in ('x', 'y', 'site', 'target')}
                 for i in range(8))


def _tv(a, b):
    return .5*(a.softmax(-1)-b.softmax(-1)).abs().sum(-1)


@torch.no_grad()
def evaluate(model, batch, device, batch_size=4):
    model.eval()
    n = len(batch[0]['x'])
    counts = {name: {'correct': 0, 'paired_top1': 0,
                     'paired_candidate': 0, 'edit_direction': 0,
                     'site_nll': 0., 'own_tv': 0.,
                     'nuisance_tv': 0.}
              for name in KINDS}
    token_loss = token_count = 0.
    for start in range(0, n, batch_size):
        rows = [{key: value[start:start+batch_size].to(device)
                 for key, value in arm.items()} for arm in batch]
        logits = [model(row['x']).float() for row in rows]
        m = len(rows[0]['x'])
        ix = torch.arange(m, device=device)
        z = [[out[ix, row['site'][:, k]] for k in range(3)]
             for out, row in zip(logits, rows)]
        correct = [[z[i][k].argmax(-1) == rows[i]['target'][:, k]
                    for k in range(3)] for i in range(8)]
        for i in range(8):
            for k, name in enumerate(KINDS):
                counts[name]['correct'] += correct[i][k].sum().item()
                counts[name]['site_nll'] += F.cross_entropy(
                    z[i][k], rows[i]['target'][:, k],
                    reduction='sum').item()
            token_loss += F.cross_entropy(
                logits[i].flatten(0, 1), rows[i]['y'].flatten(),
                ignore_index=-100, reduction='sum').item()
            token_count += (rows[i]['y'] != -100).sum().item()
        for k, name in enumerate(KINDS):
            # Every edge of the 2x2x2 cube changes exactly one factor.
            for left, arm in enumerate(ARMS):
                if arm[k] != ARMS[0][k]:
                    continue
                edited = list(arm)
                edited[k] = ARMS[-1][k]
                right = ARM_INDEX[tuple(edited)]
                target_a = rows[left]['target'][:, k]
                target_b = rows[right]['target'][:, k]
                la = z[left][k][ix, target_a]-z[left][k][ix, target_b]
                lb = z[right][k][ix, target_a]-z[right][k][ix, target_b]
                counts[name]['paired_top1'] += (
                    correct[left][k] & correct[right][k]).sum().item()
                counts[name]['paired_candidate'] += (
                    (la > 0) & (lb < 0)).sum().item()
                counts[name]['edit_direction'] += (la > lb).sum().item()
                counts[name]['own_tv'] += _tv(z[left][k],
                                               z[right][k]).sum().item()
            other = (k+1) % 3
            for left, arm in enumerate(ARMS):
                if arm[other] != ARMS[0][other]:
                    continue
                edited = list(arm)
                edited[other] = ARMS[-1][other]
                right = ARM_INDEX[tuple(edited)]
                counts[name]['nuisance_tv'] += _tv(
                    z[left][k], z[right][k]).sum().item()
    out = {'examples': n, 'token_nll': token_loss/token_count}
    for name in KINDS:
        c = counts[name]
        out[name] = {'top1': c['correct']/(8*n),
                     'paired_top1': c['paired_top1']/(4*n),
                     'candidate_paired': c['paired_candidate']/(4*n),
                     'edit_direction': c['edit_direction']/(4*n),
                     'site_nll': c['site_nll']/(8*n),
                     'own_edit_tv': c['own_tv']/(4*n),
                     'other_rule_edit_tv': c['nuisance_tv']/(4*n)}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--steps', type=int, default=500)
    p.add_argument('--train-worlds', type=int, default=1000)
    p.add_argument('--val-worlds', type=int, default=24)
    p.add_argument('--test-worlds', type=int, default=40)
    p.add_argument('--n-ctx', type=int, default=320)
    p.add_argument('--d-model', type=int, default=64)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--alias-weight', type=float, default=500.)
    p.add_argument('--bracket-weight', type=float, default=100.)
    p.add_argument('--caps-weight', type=float, default=100.)
    p.add_argument('--lr', type=float, default=.0005)
    p.add_argument('--seed', type=int, default=55301)
    p.add_argument('--save-every', type=int, default=250)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available()
                   else 'cpu')
    args = p.parse_args()
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    rng = random.Random(args.seed)
    train, train_attempts = draw(rng, args.train_worlds, 'train',
                                 tokenizer, args.n_ctx)
    val, val_attempts = draw(rng, args.val_worlds, 'heldout',
                             tokenizer, args.n_ctx, unique=True)
    test, test_attempts = draw(
        rng, args.test_worlds, 'heldout', tokenizer, args.n_ctx,
        unique=True, excluded={key for key, _ in val})
    assert not {key for key, _ in train} & {key for key, _ in val+test}
    training, validation, testing = map(stack, (train, val, test))
    xtrain = torch.cat([arm['x'] for arm in training])
    ytrain = torch.cat([arm['y'] for arm in training])
    strain = torch.cat([arm['site'] for arm in training])
    torch.manual_seed(args.seed+args.depth)
    model = DeepModel(50257, args.d_model, args.heads,
                      SPECS[f'block{args.depth}'], args.n_ctx,
                      norm='rms', residual='lerp',
                      attention='bilinear').to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    digest = hashlib.sha256(json.dumps({**vars(args), 'output': None,
        'checkpoint': None, 'steps': None, 'save_every': None,
        'resume': None}, sort_keys=True, default=str).encode()).hexdigest()
    resume_path = args.checkpoint.with_suffix('.resume.pt')
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    completed, losses = 0, []
    if args.resume and resume_path.exists():
        state = torch.load(resume_path, map_location=args.device,
                           weights_only=True)
        if state['digest'] != digest:
            raise ValueError('Resume configuration mismatch')
        model.load_state_dict(state['model'])
        opt.load_state_dict(state['optimizer'])
        completed, losses = state['completed'], state['losses']
        torch.set_rng_state(state['torch_rng'].cpu())
        if args.device.startswith('cuda'):
            torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda_rng']])
    starting_step = completed
    starting = evaluate(model, validation, args.device)

    def save():
        state = {'digest': digest, 'completed': completed, 'losses': losses,
                 'model': {k: v.detach().cpu()
                           for k, v in model.state_dict().items()},
                 'optimizer': opt.state_dict(),
                 'torch_rng': torch.get_rng_state(),
                 'cuda_rng': torch.cuda.get_rng_state_all()
                 if args.device.startswith('cuda') else []}
        tmp = resume_path.with_suffix('.tmp')
        torch.save(state, tmp)
        os.replace(tmp, resume_path)

    weights = (args.alias_weight, args.bracket_weight, args.caps_weight)
    for step in range(completed, args.steps):
        model.train()
        indices = torch.randint(len(xtrain), (args.batch_size,))
        x, y = xtrain[indices].to(args.device), ytrain[indices].to(args.device)
        sites = strain[indices].to(args.device)
        opt.zero_grad(set_to_none=True)
        logits = model(x).float()
        per = F.cross_entropy(logits.flatten(0, 1), y.flatten(),
                              ignore_index=-100, reduction='none').view_as(y)
        token_weights = (y != -100).float()
        ix = torch.arange(len(x), device=args.device)
        for k, weight in enumerate(weights):
            token_weights[ix, sites[:, k]] = weight
        loss = (per*token_weights).sum()/token_weights.sum()
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss at step {step}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
        completed = step+1
        losses.append(loss.item())
        if completed % args.save_every == 0 or completed == args.steps:
            save()
            print('step', completed, 'loss', round(loss.item(), 4),
                  flush=True)
    result = {'config': {**vars(args), 'output': str(args.output),
                         'checkpoint': str(args.checkpoint)},
              'data': {'train_worlds': len(train), 'val_worlds': len(val),
                       'test_worlds': len(test),
                       'train_attempts': train_attempts,
                       'val_attempts': val_attempts,
                       'test_attempts': test_attempts,
                       'unique_train_games': len({key for key, _ in train}),
                       'test_game_sha256': [hashlib.sha256(repr(key).encode())
                                            .hexdigest() for key, _ in test]},
              'starting_step': starting_step,
              'starting_validation': starting,
              'training': {'completed': completed,
                           'last50_loss': sum(losses[-50:])/len(losses[-50:])},
              'validation': evaluate(model, validation, args.device),
              'test': evaluate(model, testing, args.device),
              'caveat': 'Human report carrier has no full next-token oracle; '
                        'the alias coreference is surface-screened. '
                        'Semicolon/case edits need manual semantic review. '
                        'Only behaviorally competent rules warrant '
                        'weight-based interpretation.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    torch.save({'state_dict': {k: v.detach().cpu()
                               for k, v in model.state_dict().items()},
                'config': result['config']}, args.checkpoint)
    print('saved', args.output, flush=True)


if __name__ == '__main__':
    main()

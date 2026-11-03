"""Train a bilinear LM on alias and bracket rules in the same report."""

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
from sports_alias_bracket_dgp import realize, sample_plan
from train_sports_report_multirule import DeepModel, PAD, SPECS


ARMS = (('a', 'round'), ('b', 'round'), ('a', 'square'), ('b', 'square'))


def aligned(doc, tokenizer, n_ctx):
    enc = tokenizer.encode(doc.text, add_special_tokens=False)
    positions, targets = [], []
    for site in doc.sites:
        found = [i for i, (start, end) in enumerate(enc.offsets)
                 if start <= site.start < end]
        if len(found) != 1 or found[0] < 1 or found[0] >= n_ctx or \
                enc.offsets[found[0]][1] != site.end:
            return None
        positions.append(found[0]-1)
        targets.append(enc.ids[found[0]])
    length = min(len(enc.ids)-1, n_ctx, max(positions)+41)
    x = torch.full((n_ctx,), PAD, dtype=torch.long)
    y = torch.full((n_ctx,), -100, dtype=torch.long)
    x[:length] = torch.tensor(enc.ids[:length])
    y[:length] = torch.tensor(enc.ids[1:length+1])
    return {'x': x, 'y': y,
            'site': torch.tensor(positions, dtype=torch.long),
            'target': torch.tensor(targets, dtype=torch.long)}


def quad(plan, tokenizer, n_ctx):
    rows = tuple(aligned(realize(plan, *arm), tokenizer, n_ctx)
                 for arm in ARMS)
    if any(row is None for row in rows):
        return None
    targets = [row['target'].tolist() for row in rows]
    if not (targets[0][0] != targets[1][0] and
            targets[0][1] == targets[1][1] and
            targets[0][0] == targets[2][0] and
            targets[0][1] != targets[2][1] and
            targets[1][0] == targets[3][0] and
            targets[2][1] == targets[3][1]):
        return None
    return rows


def draw(rng, count, group, tokenizer, n_ctx, unique=False, excluded=(),
         alias_prior='roster', alias_exponent=0.):
    out, seen, attempts = [], set(excluded), 0
    while len(out) < count:
        attempts += 1
        if attempts > count*500:
            raise RuntimeError('Too few aligned joint-rule reports')
        plan = sample_plan(rng, group, alias_prior, alias_exponent)
        key = plan.alias.game_key
        if unique and key in seen:
            continue
        rows = quad(plan, tokenizer, n_ctx)
        if rows is None:
            continue
        out.append((key, rows))
        seen.add(key)
    return out, attempts


def stack(records):
    return tuple({key: torch.stack([rows[arm][key]
                                    for _, rows in records])
                  for key in ('x', 'y', 'site', 'target')}
                 for arm in range(4))


def _tv(a, b):
    return .5*(a.softmax(-1)-b.softmax(-1)).abs().sum(-1)


@torch.no_grad()
def evaluate(model, batch, device, batch_size=8):
    model.eval()
    n = len(batch[0]['x'])
    total = {key: 0. for key in (
        'alias_top1', 'bracket_top1', 'alias_paired_top1',
        'bracket_paired_top1', 'alias_candidate_paired',
        'bracket_candidate_paired', 'alias_edit_direction',
        'bracket_edit_direction', 'alias_target_tv',
        'alias_bracket_nuisance_tv', 'bracket_target_tv',
        'bracket_alias_nuisance_tv', 'alias_site_nll',
        'bracket_site_nll', 'token_loss', 'token_count')}
    for start in range(0, n, batch_size):
        rows = [{key: value[start:start+batch_size].to(device)
                 for key, value in arm.items()} for arm in batch]
        outputs = [model(row['x']).float() for row in rows]
        m = len(rows[0]['x'])
        ix = torch.arange(m, device=device)
        z = [[out[ix, row['site'][:, kind]] for kind in (0, 1)]
             for out, row in zip(outputs, rows)]
        correct = [[z[arm][kind].argmax(-1) == rows[arm]['target'][:, kind]
                    for kind in (0, 1)] for arm in range(4)]
        for arm in range(4):
            for kind, name in ((0, 'alias'), (1, 'bracket')):
                total[name+'_top1'] += correct[arm][kind].sum().item()
                total[name+'_site_nll'] += F.cross_entropy(
                    z[arm][kind], rows[arm]['target'][:, kind],
                    reduction='sum').item()
            total['token_loss'] += F.cross_entropy(
                outputs[arm].flatten(0, 1), rows[arm]['y'].flatten(),
                ignore_index=-100, reduction='sum').item()
            total['token_count'] += (rows[arm]['y'] != -100).sum().item()
        for left, right in ((0, 1), (2, 3)):
            total['alias_paired_top1'] += \
                (correct[left][0] & correct[right][0]).sum().item()
            a = rows[left]['target'][:, 0]
            b = rows[right]['target'][:, 0]
            good_a = z[left][0][ix, a] > z[left][0][ix, b]
            good_b = z[right][0][ix, b] > z[right][0][ix, a]
            total['alias_candidate_paired'] += (good_a & good_b).sum().item()
            total['alias_edit_direction'] += ((z[left][0][ix, a]-
                z[left][0][ix, b]) > (z[right][0][ix, a]-
                z[right][0][ix, b])).sum().item()
        for left, right in ((0, 2), (1, 3)):
            total['bracket_paired_top1'] += \
                (correct[left][1] & correct[right][1]).sum().item()
            a = rows[left]['target'][:, 1]
            b = rows[right]['target'][:, 1]
            good_a = z[left][1][ix, a] > z[left][1][ix, b]
            good_b = z[right][1][ix, b] > z[right][1][ix, a]
            total['bracket_candidate_paired'] += (good_a & good_b).sum().item()
            total['bracket_edit_direction'] += ((z[left][1][ix, a]-
                z[left][1][ix, b]) > (z[right][1][ix, a]-
                z[right][1][ix, b])).sum().item()
        for name, kind, left, right in (
                ('alias_target_tv', 0, 0, 1),
                ('alias_bracket_nuisance_tv', 0, 0, 2),
                ('bracket_target_tv', 1, 0, 2),
                ('bracket_alias_nuisance_tv', 1, 0, 1)):
            total[name] += _tv(z[left][kind], z[right][kind]).sum().item()
    output = {'examples': n,
              'token_nll': total['token_loss']/total['token_count']}
    for name in ('alias', 'bracket'):
        output[name] = {
            'top1': total[name+'_top1']/(4*n),
            'paired_top1': total[name+'_paired_top1']/(2*n),
            'candidate_paired': total[name+'_candidate_paired']/(2*n),
            'edit_direction': total[name+'_edit_direction']/(2*n),
            'site_nll': total[name+'_site_nll']/(4*n),
            'target_tv': total[name+'_target_tv']/n,
            'cross_rule_nuisance_tv': total[
                name+('_bracket' if name == 'alias' else '_alias')+
                '_nuisance_tv']/n,
        }
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--train-worlds', type=int, default=2000)
    p.add_argument('--val-worlds', type=int, default=24)
    p.add_argument('--test-worlds', type=int, default=40)
    p.add_argument('--n-ctx', type=int, default=320)
    p.add_argument('--d-model', type=int, default=64)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--alias-weight', type=float, default=500.)
    p.add_argument('--bracket-weight', type=float, default=100.)
    p.add_argument('--alias-prior', choices=('roster', 'eligible_target'),
                   default='roster')
    p.add_argument('--alias-exponent', type=float, default=0.)
    p.add_argument('--lr', type=float, default=.0005)
    p.add_argument('--seed', type=int, default=98201)
    p.add_argument('--save-every', type=int, default=500)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    if args.depth not in (1, 2, 3, 4):
        raise ValueError('Invalid depth')
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    rng = random.Random(args.seed)
    train, train_attempts = draw(
        rng, args.train_worlds, 'train', tokenizer, args.n_ctx,
        alias_prior=args.alias_prior, alias_exponent=args.alias_exponent)
    val, val_attempts = draw(
        rng, args.val_worlds, 'heldout', tokenizer, args.n_ctx, True,
        alias_prior=args.alias_prior, alias_exponent=args.alias_exponent)
    test, test_attempts = draw(
        rng, args.test_worlds, 'heldout', tokenizer, args.n_ctx, True,
        {key for key, _ in val}, args.alias_prior, args.alias_exponent)
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
            torch.cuda.set_rng_state_all([value.cpu()
                                          for value in state['cuda_rng']])
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
    for step in range(completed, args.steps):
        model.train()
        indices = torch.randint(len(xtrain), (args.batch_size,))
        x, y = xtrain[indices].to(args.device), ytrain[indices].to(args.device)
        site = strain[indices].to(args.device)
        opt.zero_grad(set_to_none=True)
        logits = model(x).float()
        per = F.cross_entropy(logits.flatten(0, 1), y.flatten(),
                              ignore_index=-100, reduction='none').view_as(y)
        weights = (y != -100).float()
        ix = torch.arange(len(x), device=args.device)
        weights[ix, site[:, 0]] = args.alias_weight
        weights[ix, site[:, 1]] = args.bracket_weight
        loss = (per*weights).sum()/weights.sum()
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss at step {step}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
        completed = step+1
        losses.append(loss.item())
        if completed % args.save_every == 0 or completed == args.steps:
            save()
            print('step', completed, 'loss', round(loss.item(), 4), flush=True)
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
              'caveat': 'Same human report carries two controlled rule '
                        'families; human prose has no full oracle and alias '
                        'coreference is surface-screened. Square brackets '
                        'are an intervention rather than a natural-form claim.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    torch.save({'state_dict': {key: value.detach().cpu()
                               for key, value in model.state_dict().items()},
                'config': result['config']}, args.checkpoint)
    print('saved', args.output, flush=True)


if __name__ == '__main__':
    main()

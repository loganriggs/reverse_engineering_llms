"""Bilinear LM baseline on natural-report surname reuse under alias edits."""

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
from sports_entity_alias_dgp import realize, sample_plan
from train_sports_report_multirule import DeepModel, PAD, SPECS


def aligned(doc, tokenizer, n_ctx):
    enc = tokenizer.encode(doc.text, add_special_tokens=False)
    found = [i for i, (start, end) in enumerate(enc.offsets)
             if start <= doc.answer_start < end]
    if len(found) != 1:
        return None
    index = found[0]
    if index < 1 or index >= n_ctx or \
            enc.offsets[index][1] != doc.answer_end:
        return None
    length = min(len(enc.ids)-1, n_ctx, index+40)
    x = torch.full((n_ctx,), PAD, dtype=torch.long)
    y = torch.full((n_ctx,), -100, dtype=torch.long)
    x[:length] = torch.tensor(enc.ids[:length])
    y[:length] = torch.tensor(enc.ids[1:length+1])
    return {'x': x, 'y': y, 'site': index-1,
            'target': enc.ids[index], 'document': doc}


def draw(rng, count, group, tokenizer, n_ctx, unique=False, excluded=(),
         alias_prior='roster', alias_exponent=0.):
    out, seen, attempts = [], set(excluded), 0
    while len(out) < count:
        attempts += 1
        if attempts > count*300:
            raise RuntimeError('Too few aligned unique reports')
        plan = sample_plan(rng, group, alias_prior=alias_prior,
                           alias_exponent=alias_exponent)
        if unique and plan.game_key in seen:
            continue
        docs = (realize(plan, 'a'), realize(plan, 'b'))
        rows = tuple(aligned(doc, tokenizer, n_ctx) for doc in docs)
        if any(row is None for row in rows) or \
                rows[0]['target'] == rows[1]['target']:
            continue
        out.append((plan.game_key, rows))
        seen.add(plan.game_key)
    return out, attempts


def stack(records):
    return tuple({key: torch.stack([row[arm][key] for _, row in records])
                  if key in ('x', 'y') else
                  torch.tensor([row[arm][key] for _, row in records])
                  for key in ('x', 'y', 'site', 'target')}
                 for arm in (0, 1))


@torch.no_grad()
def evaluate(model, data, device, batch_size=8):
    model.eval()
    n = len(data[0]['x'])
    top1_a = top1_b = paired = flip = 0
    candidate_a = candidate_b = candidate_paired = 0
    site_nll = token_loss = token_count = tv = 0.
    for start in range(0, n, batch_size):
        rows = [{k: v[start:start+batch_size].to(device)
                 for k, v in arm.items()} for arm in data]
        outputs = [model(row['x']).float() for row in rows]
        b = len(rows[0]['x'])
        ix = torch.arange(b, device=device)
        logits = [out[ix, row['site']] for out, row in zip(outputs, rows)]
        correct = [z.argmax(-1) == row['target']
                   for z, row in zip(logits, rows)]
        top1_a += correct[0].sum().item()
        top1_b += correct[1].sum().item()
        paired += (correct[0] & correct[1]).sum().item()
        a, btarget = rows[0]['target'], rows[1]['target']
        before = logits[0][ix, a]-logits[0][ix, btarget]
        after = logits[1][ix, a]-logits[1][ix, btarget]
        flip += (before > after).sum().item()
        good_a, good_b = before > 0, after < 0
        candidate_a += good_a.sum().item()
        candidate_b += good_b.sum().item()
        candidate_paired += (good_a & good_b).sum().item()
        tv += .5*(logits[0].softmax(-1)-
                  logits[1].softmax(-1)).abs().sum(-1).sum().item()
        for row, z, out in zip(rows, logits, outputs):
            site_nll += F.cross_entropy(z, row['target'],
                                       reduction='sum').item()
            token_loss += F.cross_entropy(out.flatten(0, 1),
                                          row['y'].flatten(),
                                          ignore_index=-100,
                                          reduction='sum').item()
            token_count += (row['y'] != -100).sum().item()
    return {'paired_top1': paired/n, 'top1_a': top1_a/n,
            'top1_b': top1_b/n, 'edit_logit_direction': flip/n,
            'candidate_rank_a': candidate_a/n,
            'candidate_rank_b': candidate_b/n,
            'candidate_rank_paired': candidate_paired/n,
            'answer_site_nll': site_nll/(2*n),
            'token_nll': token_loss/token_count,
            'mean_edit_tv': tv/n, 'paired_examples': n}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--train-pairs', type=int, default=2000)
    p.add_argument('--val-pairs', type=int, default=32)
    p.add_argument('--test-pairs', type=int, default=64)
    p.add_argument('--n-ctx', type=int, default=384)
    p.add_argument('--d-model', type=int, default=64)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--site-weight', type=float, default=500.)
    p.add_argument('--alias-prior', choices=('roster', 'eligible_target'),
                   default='roster')
    p.add_argument('--alias-exponent', type=float, default=0.)
    p.add_argument('--lr', type=float, default=.0005)
    p.add_argument('--seed', type=int, default=97601)
    p.add_argument('--save-every', type=int, default=500)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    if args.depth not in (1, 2, 3, 4):
        raise ValueError('Invalid depth')
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    rng = random.Random(args.seed)
    train, train_attempts = draw(rng, args.train_pairs, 'train', tokenizer,
                                 args.n_ctx, alias_prior=args.alias_prior,
                                 alias_exponent=args.alias_exponent)
    val, val_attempts = draw(rng, args.val_pairs, 'heldout', tokenizer,
                             args.n_ctx, True,
                             alias_prior=args.alias_prior,
                             alias_exponent=args.alias_exponent)
    test, test_attempts = draw(rng, args.test_pairs, 'heldout', tokenizer,
                               args.n_ctx, True,
                               {key for key, _ in val},
                               alias_prior=args.alias_prior,
                               alias_exponent=args.alias_exponent)
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
    starting_step = completed
    starting = evaluate(model, validation, args.device)
    for step in range(completed, args.steps):
        model.train()
        index = torch.randint(len(xtrain), (args.batch_size,))
        x = xtrain[index].to(args.device)
        y = ytrain[index].to(args.device)
        site = strain[index].to(args.device)
        opt.zero_grad(set_to_none=True)
        logits = model(x).float()
        per = F.cross_entropy(logits.flatten(0, 1), y.flatten(),
                              ignore_index=-100, reduction='none').view_as(y)
        weights = (y != -100).float()
        weights[torch.arange(len(x), device=args.device), site] = \
            args.site_weight
        loss = (per*weights).sum()/weights.sum()
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss at step {step}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
        losses.append(loss.item())
        completed = step+1
        if completed % args.save_every == 0 or completed == args.steps:
            save()
            print('step', step+1, 'loss', round(loss.item(), 4), flush=True)
    result = {'config': {**vars(args), 'output': str(args.output),
                         'checkpoint': str(args.checkpoint)},
              'data': {'train_pairs': len(train), 'val_pairs': len(val),
                       'test_pairs': len(test),
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
              'caveat': 'Natural report prose and inferred coreference are '
                        'not a full exact language oracle; paired alias edits '
                        'are controlled. Require behavioral competence before '
                        'claiming a recoverable weight circuit.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    torch.save({'state_dict': {key: value.detach().cpu()
                               for key, value in model.state_dict().items()},
                'config': result['config']}, args.checkpoint)
    print('saved', args.output, flush=True)


if __name__ == '__main__':
    main()

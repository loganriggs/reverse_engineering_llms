"""Train a bilinear LM on three controlled sites in natural sports reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from sports_report_multirule_dgp import (
    increment_player, increment_score, realize, sample_plan,
    template_digest,
)


OLD_REPO = Path(__file__).resolve().parents[2]/'reverse_engineer/tensor_language'
sys.modules.pop('model', None)
sys.path.insert(0, str(OLD_REPO))
from deep_model import DeepModel, SPECS  # noqa: E402


PAD = 50256
KINDS = ('winning_score', 'margin', 'player_points')


def aligned(doc, tokenizer, n_ctx):
    enc = tokenizer.encode(doc.text, add_special_tokens=False)
    if len(enc.ids)-1 > n_ctx:
        return None
    positions = []
    targets = []
    for kind in KINDS:
        site = next(site for site in doc.sites if site.kind == kind)
        found = [i for i, (start, end) in enumerate(enc.offsets)
                 if start <= site.start < end]
        if len(found) != 1:
            return None
        index = found[0]
        if index < 1 or index >= len(enc.ids)-1 or \
                enc.offsets[index][1] != site.end:
            return None
        positions.append(index-1)
        targets.append(enc.ids[index])
    x = torch.full((n_ctx,), PAD, dtype=torch.long)
    y = torch.full((n_ctx,), -100, dtype=torch.long)
    x[:len(enc.ids)-1] = torch.tensor(enc.ids[:-1])
    y[:len(enc.ids)-1] = torch.tensor(enc.ids[1:])
    return {'x': x, 'y': y,
            'site': torch.tensor(positions, dtype=torch.long),
            'target': torch.tensor(targets, dtype=torch.long)}


def triple(plan, tokenizer, n_ctx):
    docs = (realize(plan), realize(increment_score(plan)),
            realize(increment_player(plan)))
    rows = tuple(aligned(doc, tokenizer, n_ctx) for doc in docs)
    if any(row is None for row in rows):
        return None
    # Edits change exactly the intended rule answers.
    if not (rows[0]['target'][0] != rows[1]['target'][0] and
            rows[0]['target'][1] != rows[1]['target'][1] and
            rows[0]['target'][2] == rows[1]['target'][2] and
            torch.equal(rows[0]['target'][:2], rows[2]['target'][:2]) and
            rows[0]['target'][2] != rows[2]['target'][2]):
        return None
    return rows


def draw(rng, count, group, tokenizer, n_ctx, unique=False,
         excluded=(), carrier_words=100):
    records = []
    seen = set(excluded)
    attempts = 0
    while len(records) < count:
        attempts += 1
        if attempts > count*200:
            raise RuntimeError('Too few aligned distinct games')
        plan = sample_plan(rng, group, carrier_words)
        key = plan.carrier_plan.source_game_key
        if unique and key in seen:
            continue
        rows = triple(plan, tokenizer, n_ctx)
        if rows is None:
            continue
        records.append((key, rows))
        seen.add(key)
    return records, attempts


def stack(records):
    return tuple({key: torch.stack([triple[i][key] for _, triple in records])
                  for key in ('x', 'y', 'site', 'target')}
                 for i in range(3))


@torch.no_grad()
def evaluate(model, batch, device, batch_size=8):
    model.eval()
    totals = {kind: {'full_vocab_top1': 0, 'site_nll': 0.,
                     'score_edit_shift_correct': 0,
                     'player_edit_shift_correct': 0}
              for kind in KINDS}
    token_loss = 0.
    token_count = 0
    n = len(batch[0]['x'])
    for start in range(0, n, batch_size):
        end = min(start+batch_size, n)
        rows = [{key: value[start:end].to(device) for key, value in variant.items()}
                for variant in batch]
        outputs = [model(row['x']).float() for row in rows]
        b = end-start
        ix = torch.arange(b, device=device)
        for row, logits in zip(rows, outputs):
            token_loss += F.cross_entropy(logits.flatten(0, 1),
                                          row['y'].flatten(),
                                          ignore_index=-100,
                                          reduction='sum').item()
            token_count += (row['y'] != -100).sum().item()
        for j, kind in enumerate(KINDS):
            scores = [out[ix, row['site'][:, j]]
                      for out, row in zip(outputs, rows)]
            for row, logits in zip(rows, scores):
                target = row['target'][:, j]
                totals[kind]['full_vocab_top1'] += \
                    (logits.argmax(-1) == target).sum().item()
                totals[kind]['site_nll'] -= logits.log_softmax(-1)[ix, target].sum().item()
            old = rows[0]['target'][:, j]
            for edit_ix, label in ((1, 'score_edit_shift_correct'),
                                   (2, 'player_edit_shift_correct')):
                new = rows[edit_ix]['target'][:, j]
                if torch.equal(old, new):
                    continue
                before = scores[0][ix, new]-scores[0][ix, old]
                after = scores[edit_ix][ix, new]-scores[edit_ix][ix, old]
                totals[kind][label] += (after > before).sum().item()
    out = {'triplets': n, 'token_nll': token_loss/token_count, 'sites': {}}
    for kind, row in totals.items():
        out['sites'][kind] = {
            'full_vocab_top1': row['full_vocab_top1']/(3*n),
            'site_nll': row['site_nll']/(3*n),
            'score_edit_shift_correct': row['score_edit_shift_correct']/n
            if kind != 'player_points' else None,
            'player_edit_shift_correct': row['player_edit_shift_correct']/n
            if kind == 'player_points' else None,
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint-dir', type=Path, required=True)
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--train-triplets', type=int, default=2000)
    p.add_argument('--val-triplets', type=int, default=32)
    p.add_argument('--test-triplets', type=int, default=64)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--n-ctx', type=int, default=320)
    p.add_argument('--carrier-words', type=int, default=100)
    p.add_argument('--d-model', type=int, default=64)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--site-weight', type=float, default=100.)
    p.add_argument('--lr', type=float, default=.0005)
    p.add_argument('--seed', type=int, default=96701)
    p.add_argument('--save-every', type=int, default=100)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    if a.depth not in (1, 2, 3, 4) or a.site_weight < 1:
        raise ValueError('Invalid depth or site weight')
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    rng = random.Random(a.seed)
    train, train_attempts = draw(rng, a.train_triplets, 'train', tokenizer,
                                 a.n_ctx, carrier_words=a.carrier_words)
    val, val_attempts = draw(rng, a.val_triplets, 'heldout', tokenizer,
                             a.n_ctx, True, carrier_words=a.carrier_words)
    test, test_attempts = draw(rng, a.test_triplets, 'heldout', tokenizer,
                               a.n_ctx, True, {key for key, _ in val},
                               a.carrier_words)
    assert not {key for key, _ in train} & {key for key, _ in val+test}
    training, validation, testing = map(stack, (train, val, test))
    xtrain = torch.cat([row['x'] for row in training])
    ytrain = torch.cat([row['y'] for row in training])
    strain = torch.cat([row['site'] for row in training])
    torch.manual_seed(a.seed+a.depth)
    model = DeepModel(50257, a.d_model, a.heads,
                      SPECS[f'block{a.depth}'], a.n_ctx,
                      norm='rms', residual='lerp',
                      attention='bilinear').to(a.device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    digest = hashlib.sha256(json.dumps({**vars(a),
        'output': None, 'checkpoint_dir': None, 'steps': None,
        'resume': None, 'template_digest': template_digest()},
        sort_keys=True, default=str).encode()).hexdigest()
    a.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    resume_path = a.checkpoint_dir/'resume_latest.pt'
    completed = 0
    losses = []
    if a.resume and resume_path.exists():
        state = torch.load(resume_path, map_location=a.device, weights_only=True)
        if state['digest'] != digest:
            raise ValueError('Resume configuration mismatch')
        model.load_state_dict(state['model'])
        opt.load_state_dict(state['optimizer'])
        completed, losses = state['completed'], state['losses']
        torch.set_rng_state(state['torch_rng'].cpu())
        if a.device.startswith('cuda'):
            torch.cuda.set_rng_state_all([value.cpu()
                                          for value in state['cuda_rng']])
    def save():
        state = {'digest': digest, 'completed': completed, 'losses': losses,
                 'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                 'optimizer': opt.state_dict(), 'torch_rng': torch.get_rng_state(),
                 'cuda_rng': torch.cuda.get_rng_state_all()
                 if a.device.startswith('cuda') else []}
        tmp = resume_path.with_suffix('.tmp')
        torch.save(state, tmp)
        os.replace(tmp, resume_path)
    output = {'config': {**vars(a), 'output': str(a.output),
                         'checkpoint_dir': str(a.checkpoint_dir)},
              'template_digest': template_digest(),
              'data': {'train_triplets': len(train), 'val_triplets': len(val),
                       'test_triplets': len(test), 'train_attempts': train_attempts,
                       'val_attempts': val_attempts, 'test_attempts': test_attempts,
                       'unique_train_games': len({k for k, _ in train})},
              'starting_step': completed,
              'starting_validation': evaluate(model, validation, a.device),
              'caveat': 'Empirical natural carrier lacks full next-token oracle. '
                        'Only inserted sites have controlled rules. Behavior '
                        'must be gated by rule and edit before weight analysis.'}
    clock = time.time()
    for step in range(completed, a.steps):
        model.train()
        index = torch.randint(len(xtrain), (a.batch_size,))
        x = xtrain[index].to(a.device)
        y = ytrain[index].to(a.device)
        site = strain[index].to(a.device)
        opt.zero_grad(set_to_none=True)
        logits = model(x).float()
        per = F.cross_entropy(logits.flatten(0, 1), y.flatten(),
                              ignore_index=-100, reduction='none').view_as(y)
        weights = (y != -100).float()
        weights[torch.arange(len(x), device=a.device)[:, None], site] = a.site_weight
        loss = (per*weights).sum()/weights.sum()
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss at step {step}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
        losses.append(loss.item())
        completed = step+1
        if completed % a.save_every == 0 or completed == a.steps:
            save()
            print('step', completed, 'loss', round(loss.item(), 4), flush=True)
    output['training'] = {'completed': completed,
                          'last50_loss': sum(losses[-50:])/len(losses[-50:]),
                          'elapsed_seconds': round(time.time()-clock, 2)}
    output['validation'] = evaluate(model, validation, a.device)
    output['test'] = evaluate(model, testing, a.device)
    a.output.write_text(json.dumps(output, indent=2)+'\n')
    checkpoint = a.checkpoint_dir/'model.pt'
    torch.save({'state_dict': {k: v.detach().cpu()
                               for k, v in model.state_dict().items()},
                'config': output['config'], 'template_digest': template_digest()},
               checkpoint)
    print('saved', a.output, flush=True)


if __name__ == '__main__':
    main()

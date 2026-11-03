"""Train small bilinear LMs on document-split natural delimiter transforms."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


OLD_REPO = Path(__file__).resolve().parents[2] / 'reverse_engineer/tensor_language'
if not (OLD_REPO / 'deep_model.py').is_file():
    raise FileNotFoundError(OLD_REPO / 'deep_model.py')
sys.path.insert(0, str(OLD_REPO))
from deep_model import DeepModel, SPECS  # noqa: E402


def _model(vocab_size, n_ctx, depth, d_model, heads, device, norm='rms'):
    if norm not in ('rms', 'none'):
        raise ValueError(norm)
    norm_arg = False if norm == 'none' else 'rms'
    model = DeepModel(vocab_size, d_model, heads,
                      SPECS[f'block{depth}'], n_ctx,
                      norm=norm_arg, residual='lerp',
                      attention='bilinear').to(device)
    expected = ('Identity' if norm == 'none' else 'RMSNorm')
    assert all(layer.norm.__class__.__name__ == expected
               for layer in model.layers)
    return model


def _forward(model, x, device):
    return model(x)


@torch.no_grad()
def evaluate(model, paired, device, batch_size=8):
    model.eval()
    totals = dict(token_nll=0., tokens=0, round_nll=0., square_nll=0.,
                  round_top1=0, square_top1=0, round_two_way=0,
                  square_two_way=0, both_two_way=0,
                  correct_margin_shift=0., positive_shift=0,
                  pure_pairs=0, pure_both_top1=0,
                  merged_both_top1=0, pure_both_two_way=0,
                  merged_both_two_way=0, n=0)
    round_data, square_data = paired['round'], paired['square']
    n = len(round_data['x'])
    assert n == len(square_data['x'])
    for start in range(0, n, batch_size):
        end = min(n, start+batch_size)
        rr = {k: v[start:end].to(device) for k, v in round_data.items()}
        ss = {k: v[start:end].to(device) for k, v in square_data.items()}
        lr = _forward(model, rr['x'], device).float()
        ls = _forward(model, ss['x'], device).float()
        for logits, row in ((lr, rr), (ls, ss)):
            loss = F.cross_entropy(logits.flatten(0, 1), row['y'].flatten(),
                                   reduction='none', ignore_index=-100)
            totals['token_nll'] += loss.sum().item()
            totals['tokens'] += (row['y'] != -100).sum().item()
        ix = torch.arange(end-start, device=device)
        round_logits = lr[ix, rr['site_position']]
        square_logits = ls[ix, ss['site_position']]
        round_id, square_id = rr['target_id'], ss['target_id']
        assert torch.equal(round_id != square_id,
                           torch.ones_like(round_id, dtype=torch.bool))
        pr = F.log_softmax(round_logits, -1)
        ps = F.log_softmax(square_logits, -1)
        totals['round_nll'] -= pr[ix, round_id].sum().item()
        totals['square_nll'] -= ps[ix, square_id].sum().item()
        totals['round_top1'] += (round_logits.argmax(-1) == round_id).sum().item()
        totals['square_top1'] += (
            square_logits.argmax(-1) == square_id).sum().item()
        round_wins = round_logits[ix, round_id] > round_logits[ix, square_id]
        square_wins = square_logits[ix, square_id] > square_logits[ix, round_id]
        totals['round_two_way'] += round_wins.sum().item()
        totals['square_two_way'] += square_wins.sum().item()
        totals['both_two_way'] += (round_wins & square_wins).sum().item()
        pure = (rr['target_is_pure_delimiter'] &
                ss['target_is_pure_delimiter'])
        round_top1 = round_logits.argmax(-1) == round_id
        square_top1 = square_logits.argmax(-1) == square_id
        totals['pure_both_top1'] += (
            round_top1 & square_top1 & pure).sum().item()
        totals['merged_both_top1'] += (
            round_top1 & square_top1 & ~pure).sum().item()
        totals['pure_both_two_way'] += (
            round_wins & square_wins & pure).sum().item()
        totals['merged_both_two_way'] += (
            round_wins & square_wins & ~pure).sum().item()
        shift = ((square_logits[ix, square_id]-
                  square_logits[ix, round_id])-
                 (round_logits[ix, square_id]-
                  round_logits[ix, round_id]))
        totals['correct_margin_shift'] += shift.sum().item()
        totals['positive_shift'] += (shift > 0).sum().item()
        totals['pure_pairs'] += pure.sum().item()
        totals['n'] += end-start
    count = totals['n']
    pure_count = totals['pure_pairs']
    merged_count = count-pure_count
    return dict(n=count, token_nll=totals['token_nll']/totals['tokens'],
                round_site_nll=totals['round_nll']/count,
                square_site_nll=totals['square_nll']/count,
                round_top1=totals['round_top1']/count,
                square_top1=totals['square_top1']/count,
                round_two_way=totals['round_two_way']/count,
                square_two_way=totals['square_two_way']/count,
                both_two_way=totals['both_two_way']/count,
                mean_correct_margin_shift=(
                    totals['correct_margin_shift']/count),
                positive_shift_fraction=totals['positive_shift']/count,
                pure_delimiter_pairs=pure_count,
                pure_both_top1=(totals['pure_both_top1']/pure_count
                                if pure_count else None),
                merged_both_top1=(totals['merged_both_top1']/merged_count
                                  if merged_count else None),
                pure_both_two_way=(totals['pure_both_two_way']/pure_count
                                   if pure_count else None),
                merged_both_two_way=(
                    totals['merged_both_two_way']/merged_count
                    if merged_count else None))


def train(args):
    torch.set_num_threads(4)
    device = torch.device(args.device)
    data = torch.load(args.cache_path, map_location='cpu',
                      weights_only=True)
    n_ctx, vocab_size = data['n_ctx'], data['vocab_size']
    output = dict(config={**vars(args),
                          'cache_path': str(args.cache_path),
                          'output': str(args.output),
                          'checkpoint_dir': str(args.checkpoint_dir)},
                  n_ctx=n_ctx, vocab_size=vocab_size,
                  train_examples=len(data['train']['x']),
                  val_pairs=len(data['val']['round']['x']),
                  test_pairs=len(data['test']['round']['x']),
                  objective='ordinary next-token CE with optional '
                            'site-weighted CE; natural source text plus '
                            'sampled delimiter style',
                  results={})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for depth in (int(x) for x in args.depths.split(',')):
        assert 1 <= depth <= 4
        torch.manual_seed(args.seed+depth)
        model = _model(vocab_size, n_ctx, depth,
                       args.d_model, args.heads, device, args.norm)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        rng = random.Random(args.seed+10_000+depth)
        initial = evaluate(model, data['val'], device,
                           batch_size=args.eval_batch_size)
        losses = []
        start_time = time.time()
        for step in range(args.steps):
            model.train()
            indices = torch.tensor([
                rng.randrange(len(data['train']['x']))
                for _ in range(args.batch_size)], dtype=torch.long)
            x = data['train']['x'][indices].to(device)
            y = data['train']['y'][indices].to(device)
            site = data['train']['site_position'][indices].to(device)
            opt.zero_grad(set_to_none=True)
            logits = _forward(model, x, device)
            per = F.cross_entropy(logits.float().flatten(0, 1),
                                  y.flatten(), reduction='none',
                                  ignore_index=-100).view_as(y)
            weights = (y != -100).float()
            if args.site_weight != 1.:
                ix = torch.arange(len(x), device=device)
                weights[ix, site] = args.site_weight
            loss = (per*weights).sum()/weights.sum()
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite depth={depth} step={step}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            losses.append(loss.item())
            if (step+1) % args.log_every == 0:
                print(depth, step+1, round(loss.item(), 4), flush=True)
        val = evaluate(model, data['val'], device,
                       batch_size=args.eval_batch_size)
        test = evaluate(model, data['test'], device,
                        batch_size=args.eval_batch_size)
        checkpoint = args.checkpoint_dir / f'depth{depth}_site{args.site_weight:g}.pt'
        torch.save(dict(state_dict={k: v.detach().cpu() for k, v in
                                    model.state_dict().items()},
                        depth=depth, d_model=args.d_model,
                        heads=args.heads, n_ctx=n_ctx,
                        vocab_size=vocab_size,
                        attention='bilinear', norm=args.norm, residual='lerp',
                        cache_path=str(args.cache_path)), checkpoint)
        output['results'][str(depth)] = dict(
            parameters=model.n_params,
            initial_val=initial,
            train_loss_last50=sum(losses[-50:])/len(losses[-50:]),
            val=val, test=test,
            elapsed_seconds=round(time.time()-start_time, 2),
            checkpoint_path=str(checkpoint))
        args.output.write_text(json.dumps(output, indent=2)+'\n')
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache-path', type=Path, default=Path(
        '/home/loganriggs/Coding/neuron_circuits/artifacts/'
        'corpus_delimiter_pilot/dataset.pt'))
    p.add_argument('--depths', default='1,2')
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--eval-batch-size', type=int, default=8)
    p.add_argument('--d-model', type=int, default=64)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--norm', choices=['rms', 'none'], default='rms')
    p.add_argument('--lr', type=float, default=.0005)
    p.add_argument('--site-weight', type=float, default=1.)
    p.add_argument('--seed', type=int, default=71411)
    p.add_argument('--log-every', type=int, default=50)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available()
                   else 'cpu')
    p.add_argument('--checkpoint-dir', type=Path, default=Path(
        '/home/loganriggs/Coding/neuron_circuits/artifacts/'
        'corpus_delimiter_pilot/checkpoints'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.site_weight < 1:
        raise ValueError('site_weight must be >= 1')
    result = train(args)
    print('saved', args.output, result['results'].keys(), flush=True)


if __name__ == '__main__':
    main()

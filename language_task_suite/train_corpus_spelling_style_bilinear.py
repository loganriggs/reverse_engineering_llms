"""Train bilinear LMs on a sampled document-level spelling convention."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from train_corpus_delimiter_bilinear import _model


def _pair_metrics(model, paired, device, batch_size, seen_combinations=None):
    model.eval()
    us, uk = paired['us'], paired['uk']
    n = len(us['x'])
    assert n == len(uk['x'])
    totals = dict(tokens=0, token_nll=0., us_nll=0., uk_nll=0.,
                  us_top1=0, uk_top1=0, us_two_way=0,
                  uk_two_way=0, both_two_way=0, positive_shift=0,
                  signed_shift=0.)
    by_root = {}
    seen = dict(n=0, correct=0)
    unseen = dict(n=0, correct=0)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(n, start+batch_size)
            a = {k: v[start:end].to(device) for k, v in us.items()}
            b = {k: v[start:end].to(device) for k, v in uk.items()}
            la, lb = model(a['x']).float(), model(b['x']).float()
            for logits, row in ((la, a), (lb, b)):
                losses = F.cross_entropy(logits.flatten(0, 1),
                                         row['y'].flatten(),
                                         reduction='none',
                                         ignore_index=-100)
                totals['token_nll'] += losses.sum().item()
                totals['tokens'] += (row['y'] != -100).sum().item()
            ix = torch.arange(end-start, device=device)
            xa = la[ix, a['site_position']]
            xb = lb[ix, b['site_position']]
            aid, bid = a['target_id'], b['target_id']
            assert (aid != bid).all()
            totals['us_nll'] -= F.log_softmax(xa, -1)[ix, aid].sum().item()
            totals['uk_nll'] -= F.log_softmax(xb, -1)[ix, bid].sum().item()
            totals['us_top1'] += (xa.argmax(-1) == aid).sum().item()
            totals['uk_top1'] += (xb.argmax(-1) == bid).sum().item()
            ma = xa[ix, aid]-xa[ix, bid]
            mb = xb[ix, aid]-xb[ix, bid]
            correct = (ma > 0) & (mb < 0)
            shift = ma-mb
            totals['us_two_way'] += (ma > 0).sum().item()
            totals['uk_two_way'] += (mb < 0).sum().item()
            totals['both_two_way'] += correct.sum().item()
            totals['positive_shift'] += (shift > 0).sum().item()
            totals['signed_shift'] += shift.sum().item()
            for local in range(end-start):
                root = int(a['target_root_id'][local])
                row = by_root.setdefault(root, dict(n=0, correct=0))
                row['n'] += 1
                row['correct'] += int(correct[local])
                if seen_combinations is not None:
                    combo = (int(a['cue_root_id'][local]), root)
                    target = (seen if combo in seen_combinations else unseen)
                    target['n'] += 1
                    target['correct'] += int(correct[local])
    if n == 0:
        raise ValueError('No paired examples')
    result = dict(n=n, token_nll=totals['token_nll']/totals['tokens'],
                  us_site_nll=totals['us_nll']/n,
                  uk_site_nll=totals['uk_nll']/n,
                  us_top1=totals['us_top1']/n,
                  uk_top1=totals['uk_top1']/n,
                  us_two_way=totals['us_two_way']/n,
                  uk_two_way=totals['uk_two_way']/n,
                  both_two_way=totals['both_two_way']/n,
                  positive_shift_fraction=totals['positive_shift']/n,
                  mean_correct_margin_shift=totals['signed_shift']/n,
                  target_root={str(root):dict(
                      n=value['n'],
                      both_two_way=value['correct']/value['n'])
                      for root, value in by_root.items()})
    if seen_combinations is not None:
        result['seen_combinations'] = dict(
            n=seen['n'], both_two_way=(seen['correct']/seen['n']
                                      if seen['n'] else None))
        result['unseen_combinations'] = dict(
            n=unseen['n'], both_two_way=(unseen['correct']/unseen['n']
                                        if unseen['n'] else None))
    return result


def train(args):
    torch.set_num_threads(4)
    data = torch.load(args.cache_path, map_location='cpu',
                      weights_only=True)
    device = torch.device(args.device)
    train_rows = data['train']
    combinations = set(zip(train_rows['cue_root_id'].tolist(),
                           train_rows['target_root_id'].tolist()))
    output = dict(config={**vars(args),
                          'cache_path': str(args.cache_path),
                          'checkpoint_dir': str(args.checkpoint_dir),
                          'output': str(args.output)},
                  train_examples=len(train_rows['x']),
                  val_pairs=len(data['val']['us']['x']),
                  test_pairs=len(data['test']['us']['x']),
                  n_ctx=data['n_ctx'],
                  family_id=data['family_id'],
                  objective='ordinary next-token CE with site weighting '
                            'for document-wide sampled US/UK-associated '
                            'spelling convention',
                  results={})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for depth in (int(v) for v in args.depths.split(',')):
        torch.manual_seed(args.seed+depth)
        model = _model(data['vocab_size'], data['n_ctx'], depth,
                       args.d_model, args.heads, device, args.norm)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        rng = random.Random(args.seed+10_000+depth)
        initial = _pair_metrics(model, data['val'], device,
                                args.eval_batch_size, combinations)
        losses = []
        start_time = time.time()
        for step in range(args.steps):
            model.train()
            indices = torch.tensor([
                rng.randrange(len(train_rows['x']))
                for _ in range(args.batch_size)], dtype=torch.long)
            x = train_rows['x'][indices].to(device)
            y = train_rows['y'][indices].to(device)
            site = train_rows['site_position'][indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            per = F.cross_entropy(logits.float().flatten(0, 1),
                                  y.flatten(), reduction='none',
                                  ignore_index=-100).view_as(y)
            weights = (y != -100).float()
            ix = torch.arange(len(x), device=device)
            weights[ix, site] = args.site_weight
            loss = (per*weights).sum()/weights.sum()
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite depth={depth} step={step}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(loss.item())
            if (step+1) % args.log_every == 0:
                print(depth, step+1, round(loss.item(), 4), flush=True)
        val = _pair_metrics(model, data['val'], device,
                            args.eval_batch_size, combinations)
        test = _pair_metrics(model, data['test'], device,
                             args.eval_batch_size, combinations)
        checkpoint_path = args.checkpoint_dir / (
            f'depth{depth}_site{args.site_weight:g}_seed{args.seed}.pt')
        torch.save(dict(state_dict={key: value.detach().cpu()
                                    for key, value in
                                    model.state_dict().items()},
                        depth=depth, d_model=args.d_model,
                        heads=args.heads, n_ctx=data['n_ctx'],
                        vocab_size=data['vocab_size'], norm=args.norm,
                        attention='bilinear', residual='lerp',
                        cache_path=str(args.cache_path)), checkpoint_path)
        output['results'][str(depth)] = dict(
            parameters=model.n_params, initial_val=initial,
            train_loss_last50=sum(losses[-50:])/len(losses[-50:]),
            val=val, test=test,
            elapsed_seconds=round(time.time()-start_time, 2),
            checkpoint_path=str(checkpoint_path))
        args.output.write_text(json.dumps(output, indent=2)+'\n')
    print('saved', args.output, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache-path', type=Path, default=Path(
        '/home/loganriggs/Coding/neuron_circuits/artifacts/'
        'corpus_spelling_style_pilot/dataset.pt'))
    p.add_argument('--depths', default='1,2')
    p.add_argument('--steps', type=int, default=1500)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--eval-batch-size', type=int, default=2)
    p.add_argument('--d-model', type=int, default=64)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--norm', choices=['rms', 'none'], default='none')
    p.add_argument('--lr', type=float, default=.0005)
    p.add_argument('--site-weight', type=float, default=50.)
    p.add_argument('--seed', type=int, default=31017)
    p.add_argument('--log-every', type=int, default=250)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available()
                   else 'cpu')
    p.add_argument('--checkpoint-dir', type=Path, default=Path(
        '/home/loganriggs/Coding/neuron_circuits/artifacts/'
        'corpus_spelling_style_pilot/checkpoints'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.site_weight < 1:
        raise ValueError('site_weight must be >=1')
    train(args)


if __name__ == '__main__':
    main()

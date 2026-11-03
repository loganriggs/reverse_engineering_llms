"""Train bilinear next-token models on hidden-genre V3/V4 joint DGPs."""

import argparse
import collections
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from genre_document_v3 import (TOKEN2ID as V3_TOKEN2ID,
                               sample_sequence as sample_v3)
from genre_document_v4 import (TOKEN2ID as V4_TOKEN2ID,
                               sample_sequence as sample_v4)
from train_joint_document_v1 import DeepModel, SPECS


TASKS = ('genre_sentence_start', 'mixed_verb', 'claim_linked_item',
         'source_answer', 'section_number', 'bracket_close',
         'step_number', 'arithmetic_answer')


def make_data(n, seed, length, regime, keep_oracle, token2id, sampler):
    samples, rows = [], []
    for i in range(n):
        words, labels, distributions, genre = sampler(seed+i, length,
                                                      regime)
        rows.append([token2id[word] for word in words])
        if keep_oracle:
            samples.append((words, labels, distributions, genre))
    ids = torch.tensor(rows, dtype=torch.long)
    return samples, (ids[:, :-1], ids[:, 1:]), {tuple(row) for row in rows}


@torch.no_grad()
def evaluate(model, samples, data, device, token2id, batch_size=64):
    model.eval()
    aggregates = collections.defaultdict(lambda: collections.Counter())
    total_nll, total_tokens = 0., 0
    for start in range(0, len(samples), batch_size):
        end = min(start+batch_size, len(samples))
        x, y = (array[start:end].to(device) for array in data)
        logits = model(x)
        logp = F.log_softmax(logits, dim=-1)
        total_nll -= logp.gather(-1, y.unsqueeze(-1)).sum().item()
        total_tokens += y.numel()
        for j, (words, labels, distributions, genre) in enumerate(samples[start:end]):
            for i in range(1, len(words)):
                family = labels[i]
                if family not in TASKS:
                    continue
                q = distributions[i]
                entropy = -sum(p*math.log(p) for p in q.values())
                cross = -sum(p*logp[j, i-1, token2id[word]].item()
                             for word, p in q.items())
                for key in (family, f'{genre}:{family}'):
                    item = aggregates[key]
                    item['n'] += 1
                    item['kl'] += cross-entropy
                    item['target_nll'] -= logp[j, i-1, token2id[words[i]]].item()
                    if len(q) == 1:
                        item['deterministic'] += 1
                        item['hits'] += int(logits[j, i-1].argmax().item() ==
                                            token2id[words[i]])
    return dict(token_nll=total_nll/total_tokens,
                by_site={key: dict(n=int(row['n']),
                                   exact_oracle_kl=row['kl']/row['n'],
                                   observed_target_nll=row['target_nll']/row['n'],
                                   deterministic_n=int(row['deterministic']),
                                   deterministic_full_vocab_accuracy=(
                                       row['hits']/row['deterministic']
                                       if row['deterministic'] else None))
                         for key, row in aggregates.items()})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--generator', choices=('v3', 'v4'), default='v3')
    p.add_argument('--depths', default='1,2,3,4')
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--train-n', type=int, default=8000)
    p.add_argument('--eval-n', type=int, default=1000)
    p.add_argument('--length', type=int, default=256)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--d-model', type=int, default=64)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--seed', type=int, default=39001)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--save-models', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(4)
    token2id, sampler = ((V3_TOKEN2ID, sample_v3) if args.generator == 'v3'
                         else (V4_TOKEN2ID, sample_v4))
    _, train_data, train_text = make_data(args.train_n, args.seed, args.length,
                                          'task_rich', False, token2id, sampler)
    task_eval, task_data, task_text = make_data(args.eval_n,
                                                args.seed+100000, args.length,
                                                'task_rich', True, token2id, sampler)
    ref_eval, ref_data, ref_text = make_data(args.eval_n,
                                             args.seed+200000, args.length,
                                             'reference', True, token2id, sampler)
    output = dict(config=vars(args) | {'output': str(args.output)},
                  n_vocab=len(token2id), n_ctx=args.length-1,
                  exact_document_overlap=dict(task_rich=len(train_text & task_text),
                                              reference=len(train_text & ref_text)),
                  objective='full next-token CE; no task/genre labels in training',
                  caveat='One seed/depth, no whole-task holdout; structural '
                         'genre pilot still fails FineWeb lexical/form gate.',
                  results={})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for depth in [int(word) for word in args.depths.split(',')]:
        assert 1 <= depth <= 4
        torch.manual_seed(args.seed+depth)
        model = DeepModel(len(token2id), args.d_model, args.heads,
                          SPECS[f'block{depth}'], args.length-1,
                          norm='rms', residual='lerp', attention='bilinear').to(args.device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        rng = random.Random(args.seed+300000+depth)
        losses = []
        start = time.time()
        for step in range(args.steps):
            index = torch.tensor(rng.sample(range(args.train_n), args.batch_size))
            x, y = (array[index].to(args.device) for array in train_data)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
            if not torch.isfinite(loss):
                raise RuntimeError(f'nonfinite depth{depth} step{step}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            losses.append(loss.item())
            if (step+1) % 100 == 0:
                print(depth, step+1, round(loss.item(), 4), flush=True)
        scores = dict(task_rich=evaluate(model, task_eval, task_data,
                                        args.device, token2id),
                      reference=evaluate(model, ref_eval, ref_data,
                                         args.device, token2id))
        output['results'][str(depth)] = dict(parameters=model.n_params,
                                             train_loss_last50=sum(losses[-50:])/len(losses[-50:]),
                                             evaluations=scores,
                                             seconds=round(time.time()-start, 2))
        args.output.write_text(json.dumps(output, indent=2)+'\n')
        if args.save_models:
            torch.save(dict(state_dict={k: v.detach().cpu() for k, v in
                                        model.state_dict().items()},
                            vocabulary=token2id, depth=depth,
                            n_ctx=args.length-1,
                            model_kwargs=dict(d_model=args.d_model,
                                              n_head=args.heads, norm='rms',
                                              residual='lerp', attention='bilinear'),
                            config=output['config'], evaluations=scores),
                       args.output.with_name(args.output.stem+f'_block{depth}.pt'))
    print('saved', args.output, flush=True)


if __name__ == '__main__':
    main()

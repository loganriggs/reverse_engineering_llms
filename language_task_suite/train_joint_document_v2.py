"""Bilinear full-CE training on V2 contrastive source/task-rich documents."""

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from joint_document_v1 import TOKEN2ID
from joint_document_v2 import sample_sequence
from train_joint_document_v1 import DeepModel, SPECS, evaluate


def make_data(n, seed, length, regime):
    samples = [sample_sequence(seed+i, length, regime) for i in range(n)]
    ids = torch.tensor([[TOKEN2ID[word] for word in words]
                        for words, _, _ in samples], dtype=torch.long)
    return samples, (ids[:, :-1], ids[:, 1:])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--depths', default='1,2,3,4')
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--train-n', type=int, default=8000)
    p.add_argument('--eval-n', type=int, default=1000)
    p.add_argument('--length', type=int, default=256)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--d-model', type=int, default=64)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--seed', type=int, default=27001)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--save-models', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(4)
    train, train_data = make_data(args.train_n, args.seed,
                                  args.length, 'task_rich')
    task_eval, task_data = make_data(args.eval_n, args.seed+100000,
                                     args.length, 'task_rich')
    reference_eval, reference_data = make_data(args.eval_n, args.seed+200000,
                                               args.length, 'reference')
    train_text = {tuple(words) for words, _, _ in train}
    overlap = {kind: sum(tuple(words) in train_text for words, _, _ in docs)
               for kind, docs in [('task_rich', task_eval),
                                  ('reference', reference_eval)]}
    output = dict(config=vars(args) | {'output': str(args.output)},
                  n_vocab=len(TOKEN2ID), n_ctx=args.length-1,
                  exact_document_overlap=overlap,
                  objective='full next-token CE on task-rich sampled stream; '
                            'no supervised task loss',
                  caveat='Reference-rate and task-rich regimes are both '
                         'synthetic; neither passes FineWeb natural-form '
                         'gate. No whole-task training holdout.',
                  results={})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for depth in [int(word) for word in args.depths.split(',')]:
        assert 1 <= depth <= 4
        torch.manual_seed(args.seed+depth)
        model = DeepModel(len(TOKEN2ID), args.d_model, args.heads,
                          SPECS[f'block{depth}'], args.length-1,
                          norm='rms', residual='lerp', attention='bilinear').to(args.device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        rng = random.Random(args.seed+300000+depth)
        losses = []
        start = time.time()
        model.train()
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
        evaluations = {
            'task_rich': evaluate(model, task_eval, task_data, args.device),
            'reference': evaluate(model, reference_eval, reference_data, args.device),
        }
        output['results'][str(depth)] = dict(parameters=model.n_params,
                                             train_loss_last50=sum(losses[-50:])/len(losses[-50:]),
                                             evaluations=evaluations,
                                             seconds=round(time.time()-start, 2))
        args.output.write_text(json.dumps(output, indent=2)+'\n')
        if args.save_models:
            torch.save(dict(state_dict={k: v.detach().cpu() for k, v in
                                        model.state_dict().items()},
                            vocabulary=TOKEN2ID, depth=depth,
                            n_ctx=args.length-1,
                            model_kwargs=dict(d_model=args.d_model,
                                              n_head=args.heads, norm='rms',
                                              residual='lerp', attention='bilinear'),
                            config=output['config'], evaluations=evaluations),
                       args.output.with_name(args.output.stem+f'_block{depth}.pt'))
    print('saved', args.output, flush=True)


if __name__ == '__main__':
    main()

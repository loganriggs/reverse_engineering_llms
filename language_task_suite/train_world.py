"""Train 1–4-block bilinear models on variable-length latent-world documents."""

from __future__ import annotations

import argparse
import importlib
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from train import DeepModel, SPECS, encode, encode_full_lm, evaluate, vocabulary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--depths', default='1,2,3,4')
    p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--train-n', type=int, default=20000)
    p.add_argument('--test-n', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--d-model', type=int, default=48)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--objective', choices=('target', 'full_lm'), default='target')
    p.add_argument('--world-version', choices=('v1', 'v2'), default='v1')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output', default='language_task_suite/world_pilot.json')
    p.add_argument('--save-models', action='store_true')
    a = p.parse_args()
    sample_world = importlib.import_module('document_world' if a.world_version == 'v1'
                                          else 'document_world_v2').sample_world
    torch.set_num_threads(4)
    device = torch.device(a.device)
    train = sample_world(a.train_n, a.seed, genres=(0, 1))
    iid = sample_world(a.test_n, a.seed+1, genres=(0, 1))
    composition = sample_world(a.test_n, a.seed+2, split='composition', genres=(0, 1))
    genre = sample_world(a.test_n, a.seed+3, genres=(2,))
    splits = dict(iid=iid, composition=composition, held_genre=genre)
    vocab = vocabulary(train + iid + composition + genre)
    all_examples = train + iid + composition + genre
    n_ctx = max(len(ex.prefix) for ex in all_examples)
    def enc(examples):
        data = encode(examples, vocab, device)
        return (F.pad(data[0], (0, n_ctx-data[0].shape[1])), *data[1:])
    train_data = enc(train)
    test_data = {name: enc(examples) for name, examples in splits.items()}
    full_data = encode_full_lm(train, vocab, device, n_ctx) if a.objective == 'full_lm' else None
    train_prefixes = {ex.prefix for ex in train}
    output = dict(config=vars(a), n_vocab=len(vocab), n_ctx=n_ctx,
                  exact_prefix_overlap={name: sum(ex.prefix in train_prefixes for ex in examples)/len(examples)
                                        for name, examples in splits.items()},
                  split_definitions=dict(iid='new worlds, seen object/final-answer combinations and genres',
                      composition='held-out queried-object/final-answer hash fold, seen genres',
                      held_genre='third lexicalization, seen combinations'),
                  results={})
    path = Path(a.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    for depth in [int(x) for x in a.depths.split(',')]:
        assert 1 <= depth <= 4
        torch.manual_seed(a.seed+depth)
        model = DeepModel(len(vocab), a.d_model, a.heads, SPECS[f'block{depth}'],
                          n_ctx, norm='rms', residual='lerp', attention='bilinear').to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
        rng = random.Random(a.seed+100+depth)
        losses = []
        start = time.time()
        model.train()
        for step in range(a.steps):
            idx = torch.tensor(rng.sample(range(len(train)), a.batch_size), device=device)
            opt.zero_grad(set_to_none=True)
            if full_data is None:
                x, last, y = (train_data[0][idx], train_data[1][idx], train_data[2][idx])
                scores = model(x)[torch.arange(len(idx), device=device), last]
                loss = F.cross_entropy(scores, y)
            else:
                x, labels = full_data[0][idx], full_data[1][idx]
                scores = model(x)
                loss = F.cross_entropy(scores.flatten(0, 1), labels.flatten(), ignore_index=-100)
            if not torch.isfinite(loss):
                raise RuntimeError(f'nonfinite loss depth={depth} step={step}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
            if (step+1) % 100 == 0:
                print(f'depth={depth} step={step+1} loss={loss.item():.3f}', flush=True)
        model.eval()
        result = dict(parameters=model.n_params,
                      train_loss_last50=sum(losses[-50:])/len(losses[-50:]),
                      eval={name: evaluate(model, data) for name, data in test_data.items()},
                      seconds=round(time.time()-start, 2))
        output['results'][str(depth)] = result
        path.write_text(json.dumps(output, indent=2)+'\n')
        if a.save_models:
            ckpt = path.with_name(path.stem+f'_block{depth}.pt')
            torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                            vocabulary=vocab, depth=depth, n_ctx=n_ctx,
                            model_kwargs=dict(d_model=a.d_model, n_head=a.heads,
                                              norm='rms', residual='lerp', attention='bilinear'),
                            training_config=vars(a), metrics=result), ckpt)
        print(json.dumps(dict(depth=depth, result=result)), flush=True)
    print(f'Saved {path}')


if __name__ == '__main__':
    main()

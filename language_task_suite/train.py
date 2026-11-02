"""Train the original bilinear DeepModel on rule-labeled next-token probes.

This is *target-token supervision*, not natural-language next-token LM training.
Variant 1 is a held-out structural formulation of the same rule families.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from grammar import GENERATORS, sample

OLD_REPO = Path(os.environ.get('TENSOR_LANGUAGE_REPO',
    '/home/loganriggs/Coding/reverse_engineer/tensor_language'))
if not (OLD_REPO / 'deep_model.py').is_file():
    raise FileNotFoundError(
        f'{OLD_REPO}/deep_model.py not found; set TENSOR_LANGUAGE_REPO to your clone')
sys.path.insert(0, str(OLD_REPO))
from deep_model import DeepModel, SPECS  # noqa: E402


def vocabulary(examples):
    words = {'<PAD>'}
    for ex in examples:
        words.update(ex.prefix)
        words.add(ex.target)
        words.update(ex.candidates)
    return {word: i for i, word in enumerate(sorted(words))}


def encode(examples, vocab, device):
    length = max(len(ex.prefix) for ex in examples)
    tokens = torch.zeros(len(examples), length, dtype=torch.long)
    last = torch.empty(len(examples), dtype=torch.long)
    target = torch.empty(len(examples), dtype=torch.long)
    candidates = []
    families = []
    for i, ex in enumerate(examples):
        tokens[i, :len(ex.prefix)] = torch.tensor([vocab[w] for w in ex.prefix])
        last[i] = len(ex.prefix) - 1
        target[i] = vocab[ex.target]
        candidates.append([vocab[w] for w in ex.candidates])
        families.append(ex.family)
    return tokens.to(device), last.to(device), target.to(device), candidates, families


def encode_full_lm(examples, vocab, device, n_ctx):
    inputs = torch.zeros(len(examples), n_ctx, dtype=torch.long)
    labels = torch.full((len(examples), n_ctx), -100, dtype=torch.long)
    for i, ex in enumerate(examples):
        sequence = ex.prefix + (ex.target,)
        ids = [vocab[w] for w in sequence]
        inputs[i, :len(ex.prefix)] = torch.tensor(ids[:-1])
        labels[i, :len(ex.prefix)] = torch.tensor(ids[1:])
    return inputs.to(device), labels.to(device)


@torch.no_grad()
def evaluate(model, encoded, batch_size=256):
    x, last, y, candidates, families = encoded
    hits = {family: [0, 0, 0] for family in GENERATORS}
    total_loss = 0.0
    for start in range(0, len(y), batch_size):
        end = min(start + batch_size, len(y))
        scores = model(x[start:end])
        scores = scores[torch.arange(end-start, device=x.device), last[start:end]]
        total_loss += F.cross_entropy(scores, y[start:end], reduction='sum').item()
        pred = scores.argmax(-1).tolist()
        for j, f in enumerate(families[start:end]):
            row = start + j
            cand = candidates[row]
            choice = cand[scores[j, cand].argmax().item()]
            hits[f][0] += int(pred[j] == y[row].item())
            hits[f][1] += int(choice == y[row].item())
            hits[f][2] += 1
    return dict(loss=total_loss / len(y),
                full_accuracy=sum(v[0] for v in hits.values()) / len(y),
                choice_accuracy=sum(v[1] for v in hits.values()) / len(y),
                by_family={f: dict(full=v[0]/v[2], choice=v[1]/v[2], n=v[2])
                           for f,v in hits.items() if v[2]})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--depths', default='1,2,3,4')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--batch-size', type=int, default=48)
    p.add_argument('--train-n', type=int, default=12000)
    p.add_argument('--test-n', type=int, default=2400)
    p.add_argument('--d-model', type=int, default=48)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--families', default=','.join(GENERATORS))
    p.add_argument('--train-variants', default='0', choices=('0', '0,1'))
    p.add_argument('--save-models', action='store_true')
    p.add_argument('--objective', choices=('target', 'full_lm'), default='target')
    p.add_argument('--output', default='language_task_suite/pilot_results.json')
    a = p.parse_args()
    torch.set_num_threads(4)
    device = torch.device(a.device)
    families = tuple(a.families.split(','))
    assert all(f in GENERATORS for f in families)
    if a.train_variants == '0,1':
        train = sample(a.train_n//2, a.seed, variant=0, families=families)
        train += sample(a.train_n-a.train_n//2, a.seed+10, variant=1, families=families)
        random.Random(a.seed).shuffle(train)
    else:
        train = sample(a.train_n, a.seed, variant=0, families=families)
    iid = sample(a.test_n, a.seed + 1, variant=0, families=families)
    shift = sample(a.test_n, a.seed + 2, variant=1, families=families)
    vocab = vocabulary(train + iid + shift)
    train_data = encode(train, vocab, device)
    iid_data = encode(iid, vocab, device)
    shift_data = encode(shift, vocab, device)
    n_ctx = max(train_data[0].shape[1], iid_data[0].shape[1], shift_data[0].shape[1])
    # Pad all splits to the same width so positional buffers and shapes match.
    def padded(data):
        x, *rest = data
        return (F.pad(x, (0, n_ctx-x.shape[1])), *rest)
    train_data, iid_data, shift_data = map(padded, (train_data, iid_data, shift_data))
    full_data = encode_full_lm(train, vocab, device, n_ctx) if a.objective == 'full_lm' else None
    train_prefixes = {ex.prefix for ex in train}
    output = dict(config=vars(a), n_vocab=len(vocab), n_ctx=n_ctx,
                  objective=('last-prefix-token full-vocabulary cross entropy' if a.objective == 'target'
                             else 'full next-token cross entropy on generated sequences'),
                  split=('iid generated independently; variant1 '+
                         ('included in training' if a.train_variants == '0,1'
                          else 'holds out each family\'s surface/structural formulation')),
                  prefix_overlap=dict(iid=sum(ex.prefix in train_prefixes for ex in iid)/len(iid),
                                      variant1=sum(ex.prefix in train_prefixes for ex in shift)/len(shift)),
                  results={})
    for depth in [int(x) for x in a.depths.split(',')]:
        assert 1 <= depth <= 4
        torch.manual_seed(a.seed + depth)
        model = DeepModel(len(vocab), a.d_model, a.heads,
                          SPECS[f'block{depth}'], n_ctx,
                          norm='rms', residual='lerp', attention='bilinear').to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
        rng = random.Random(a.seed + depth)
        start = time.time()
        model.train()
        losses = []
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
                raise RuntimeError(f'nonfinite loss at depth {depth}, step {step}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
            if (step+1) % 50 == 0:
                print(f'depth={depth} step={step+1} loss={loss.item():.3f}', flush=True)
        model.eval()
        result = dict(parameters=model.n_params, train_loss_last50=sum(losses[-50:])/len(losses[-50:]),
                      iid=evaluate(model, iid_data), structural_shift=evaluate(model, shift_data),
                      seconds=round(time.time()-start, 2))
        output['results'][str(depth)] = result
        print(json.dumps(dict(depth=depth, **result)), flush=True)
        path = Path(a.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, indent=2) + '\n')
        if a.save_models:
            model_path = path.with_name(path.stem + f'_block{depth}.pt')
            torch.save(dict(state_dict={k: v.detach().cpu() for k,v in model.state_dict().items()},
                            vocabulary=vocab, depth=depth, n_ctx=n_ctx,
                            model_kwargs=dict(d_model=a.d_model, n_head=a.heads,
                                              norm='rms', residual='lerp',
                                              attention='bilinear'),
                            training_config=vars(a), metrics=result), model_path)
    print(f'Saved {a.output}')


if __name__ == '__main__':
    main()

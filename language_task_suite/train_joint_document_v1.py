"""Train original bilinear blocks on the exact-oracle joint document pilot."""

import argparse
import collections
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

OLD_REPO = Path(__file__).resolve().parents[2] / 'reverse_engineer/tensor_language'
sys.path.insert(0, str(OLD_REPO))
from deep_model import DeepModel, SPECS  # noqa: E402
from joint_document_v1 import TOKEN2ID, sample_sequence  # noqa: E402


def make_data(n, seed, length):
    samples = [sample_sequence(seed+i, length) for i in range(n)]
    ids = torch.tensor([[TOKEN2ID[w] for w in words] for words, _, _ in samples],
                       dtype=torch.long)
    return samples, (ids[:, :-1], ids[:, 1:])


@torch.no_grad()
def evaluate(model, samples, data, device, batch_size=64):
    model.eval()
    rows = collections.defaultdict(lambda: dict(n=0, nll=0., entropy=0.,
                                                 kl=0., hits=0, deterministic=0))
    total_loss, total_tokens = 0., 0
    for start in range(0, len(samples), batch_size):
        end = min(start+batch_size, len(samples))
        x, y = (tensor[start:end].to(device) for tensor in data)
        logits = model(x)
        logp = F.log_softmax(logits, dim=-1)
        total_loss += -logp.gather(-1, y.unsqueeze(-1)).sum().item()
        total_tokens += y.numel()
        for j, (words, labels, distributions) in enumerate(samples[start:end]):
            for i in range(1, len(words)):
                family = labels[i]
                if family not in ('mixed_verb', 'claim_linked_item',
                                  'source_answer', 'section_number',
                                  'bracket_close'):
                    continue
                q = distributions[i]
                record = rows[family]
                h = -sum(p*math.log(p) for p in q.values())
                cross = -sum(p*logp[j, i-1, TOKEN2ID[word]].item()
                             for word, p in q.items())
                record['n'] += 1
                record['nll'] += -logp[j, i-1, TOKEN2ID[words[i]]].item()
                record['entropy'] += h
                record['kl'] += cross-h
                if len(q) == 1:
                    record['deterministic'] += 1
                    record['hits'] += int(logits[j, i-1].argmax().item() ==
                                          TOKEN2ID[words[i]])
    return dict(token_nll=total_loss/total_tokens,
                by_site={family:dict(n=record['n'],
                                     observed_target_nll=record['nll']/record['n'],
                                     oracle_entropy=record['entropy']/record['n'],
                                     exact_oracle_kl=record['kl']/record['n'],
                                     deterministic_n=record['deterministic'],
                                     deterministic_full_vocab_accuracy=(
                                         record['hits']/record['deterministic']
                                         if record['deterministic'] else None))
                         for family, record in rows.items()})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--depths', default='1,2,3,4')
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--train-n', type=int, default=8000)
    p.add_argument('--test-n', type=int, default=1000)
    p.add_argument('--length', type=int, default=256)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--d-model', type=int, default=64)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--seed', type=int, default=26001)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--save-models', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(4)
    train, train_data = make_data(args.train_n, args.seed, args.length)
    test, test_data = make_data(args.test_n, args.seed+100000, args.length)
    train_text = {tuple(words) for words, _, _ in train}
    overlap = sum(tuple(words) in train_text for words, _, _ in test)
    output = dict(config=vars(args) | {'output': str(args.output)},
                  n_vocab=len(TOKEN2ID), n_ctx=args.length-1,
                  exact_document_overlap=overlap,
                  objective='full next-token CE; no task loss or labels in training',
                  caveat='Structural joint-DGP pilot fails FineWeb lexical/form gate; '
                         'independent seeds are not whole-task holdouts.',
                  results={})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for depth in [int(d) for d in args.depths.split(',')]:
        assert 1 <= depth <= 4
        torch.manual_seed(args.seed+depth)
        model = DeepModel(len(TOKEN2ID), args.d_model, args.heads,
                          SPECS[f'block{depth}'], args.length-1,
                          norm='rms', residual='lerp', attention='bilinear').to(args.device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        rng = random.Random(args.seed+200000+depth)
        losses = []
        start = time.time()
        model.train()
        for step in range(args.steps):
            indices = torch.tensor(rng.sample(range(args.train_n), args.batch_size))
            x, y = (array[indices].to(args.device) for array in train_data)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
            if not torch.isfinite(loss):
                raise RuntimeError(f'nonfinite loss depth{depth} step{step}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            losses.append(loss.item())
            if (step+1) % 100 == 0:
                print(depth, step+1, round(loss.item(), 4), flush=True)
        evaluation = evaluate(model, test, test_data, args.device)
        output['results'][str(depth)] = dict(parameters=model.n_params,
                                             train_loss_last50=sum(losses[-50:])/len(losses[-50:]),
                                             evaluation=evaluation,
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
                            config=output['config'], evaluation=evaluation),
                       args.output.with_name(args.output.stem+f'_block{depth}.pt'))
    print('saved', args.output, flush=True)


if __name__ == '__main__':
    main()

"""Train bilinear LMs on audited multi-genre language and score exact oracle KL.

Evaluation keeps dependency labels separate from easy frame tokens. The genre
header is part of the prefix, so all oracle distributions are observable.
"""

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from multi_genre_language import GENRES, MultiGenreLanguage, VOCAB


TASK_LABELS = ('report_mixed_verb', 'policy_mixed_action',
               'faq_answer_copy', 'faq_place_recall', 'faq_new_place',
               'list_number')


def make_batch(gen, n, length, device):
    ids, _, _ = gen.sample_batch(n, length+1)
    z = torch.tensor(ids, dtype=torch.long, device=device)
    return z[:, :-1], z[:, 1:]


def make_eval(gen, n, length, device):
    ids, labels, qs = gen.sample_batch(n, length+1)
    z = torch.tensor(ids, dtype=torch.long, device=device)
    q = np.zeros((n, length, len(VOCAB)), dtype=np.float32)
    masks = {name: np.zeros((n, length), dtype=bool)
             for name in ('all',) + GENRES + TASK_LABELS}
    masks['all'][:] = True
    for b, (row, labs, distributions) in enumerate(zip(ids, labels, qs)):
        genre = gen.id2token[row[0]]
        masks[genre][b] = True
        for t in range(1, len(row)):
            for token, p in distributions[t].items():
                q[b, t-1, gen.token2id[token]] = p
            if labs[t] in TASK_LABELS:
                masks[labs[t]][b, t-1] = True
    assert np.allclose(q.sum(-1), 1)
    return dict(x=z[:, :-1], y=z[:, 1:], q=torch.from_numpy(q).to(device),
                masks={name: torch.from_numpy(mask).to(device)
                       for name, mask in masks.items()})


@torch.no_grad()
def evaluate(model, data):
    logits = torch.cat([model(batch)[0].float()
                        for batch in data['x'].split(32)])
    logp = F.log_softmax(logits, -1)
    q = data['q']
    entropy = -(q*q.clamp_min(1e-30).log()).sum(-1)
    kl = -(q*logp).sum(-1)-entropy
    sampled_ce = -logp.gather(-1, data['y'].unsqueeze(-1)).squeeze(-1)
    out = {}
    for name, mask in data['masks'].items():
        count = int(mask.sum())
        out[name] = dict(n=count,
                         kl=float(kl[mask].mean()) if count else None,
                         oracle_entropy=float(entropy[mask].mean()) if count else None,
                         sampled_ce=float(sampled_ce[mask].mean()) if count else None)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--depths', default='1,2,3,4')
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--length', type=int, default=71)
    p.add_argument('--width', type=int, default=64)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--rmsnorm', action='store_true')
    p.add_argument('--lr', type=float, default=.003)
    p.add_argument('--neval', type=int, default=256)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cpu')
    p.add_argument('--output-dir', required=True)
    a = p.parse_args()
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config

    torch.set_num_threads(4)
    device = torch.device(a.device)
    output_dir = Path(a.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    test = make_eval(MultiGenreLanguage(seed=9101), a.neval, a.length, device)
    records = []
    for depth in [int(d) for d in a.depths.split(',')]:
        torch.manual_seed(9200+a.seed)
        config = Config(vocab_size=len(VOCAB), n_layer=depth,
                        n_head=a.heads, n_embd=a.width, seq_len=a.length,
                        block_has_mlp=(False,)*depth,
                        use_rmsnorm=a.rmsnorm)
        model = BilinearGPT(config).to(device)
        gen = MultiGenreLanguage(seed=9300+a.seed)
        optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr)
        t0 = time.time()
        snapshots = []
        model.train()
        for step in range(a.steps):
            x, y = make_batch(gen, a.batch, a.length, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)[0]
            loss = F.cross_entropy(logits.reshape(-1, len(VOCAB)), y.reshape(-1))
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss at depth {depth}, step {step}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            if (step+1) % 500 == 0 or step+1 == a.steps:
                model.eval()
                scores = evaluate(model, test)
                snapshots.append(dict(step=step+1, train_ce=float(loss.detach()),
                                      scores=scores, elapsed=time.time()-t0))
                print('depth', depth, 'step', step+1,
                      {name: round(scores[name]['kl'], 4) if scores[name]['kl'] is not None else None
                       for name in TASK_LABELS},
                      'elapsed', round(time.time()-t0, 1), flush=True)
                model.train()
                (output_dir/'results.json').write_text(json.dumps(dict(
                    config=vars(a), vocabulary=list(VOCAB), records=records,
                    current=dict(depth=depth, snapshots=snapshots)), indent=2)+'\n')
        torch.save(model.state_dict(), output_dir/f'model_depth{depth}.pt')
        records.append(dict(depth=depth, model_config=asdict(config),
                            parameters=model.count_params(), snapshots=snapshots))
        (output_dir/'results.json').write_text(json.dumps(dict(
            config=vars(a), vocabulary=list(VOCAB), records=records), indent=2)+'\n')


if __name__ == '__main__':
    main()

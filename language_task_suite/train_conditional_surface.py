"""Train bilinear LMs on visible rule-strength modes of one readable DGP."""

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from conditional_surface_language import (MODES, VOCAB,
                                          ConditionalSurfaceLanguage)
from surface_mixture_language import _history_and_clause


def make_batch(gen, n, length, device, return_verb_mask=False,
               return_verb_q=False, return_longlag=False):
    ids, labels, distributions = gen.sample_batch(n, length+1)
    z = torch.tensor(ids, dtype=torch.long, device=device)
    if return_verb_mask:
        verb = torch.tensor([[label == 'mixed_verb' for label in row[1:]]
                             for row in labels], dtype=torch.bool, device=device)
        extras = [verb]
        if return_verb_q:
            q = torch.zeros((n, length, len(VOCAB)), device=device)
            for b, (labs, dists) in enumerate(zip(labels, distributions)):
                for t in range(1, len(labs)):
                    if labs[t] == 'mixed_verb':
                        for token, mass in dists[t].items():
                            q[b, t-1, gen.token2id[token]] = mass
            assert torch.allclose(q.sum(-1)[verb], torch.ones_like(q.sum(-1)[verb]))
            extras.append(q)
        if return_longlag:
            longlag = torch.zeros((n, length), dtype=torch.bool, device=device)
            for b, (row, labs) in enumerate(zip(ids, labels)):
                tokens = [gen.id2token[i] for i in row]
                for t in range(2, len(row)):
                    if labs[t] != 'mixed_verb':
                        continue
                    body = tokens[2:t]
                    history, clause = _history_and_clause(body)
                    matches = [i for i, old in enumerate(history)
                               if old[2] == clause[2]]
                    if not matches:
                        continue
                    prior_verb = sum(len(old) for old in history[:matches[-1]])+3
                    longlag[b, t-1] = len(body)-prior_verb > 32
            extras.append(longlag)
        return (z[:, :-1], z[:, 1:], *extras)
    return z[:, :-1], z[:, 1:]


def make_eval(gen, n, length, device):
    ids, labels, qs = gen.sample_batch(n, length+1)
    z = torch.tensor(ids, dtype=torch.long, device=device)
    q = np.zeros((n, length, len(VOCAB)), dtype=np.float32)
    masks = {name: np.zeros((n, length), dtype=bool)
             for name in ('all', 'verb', 'verb_induction',
                          'verb_no_induction', 'conflict', 'frame')}
    masks['all'][:] = True
    for b, (row, labs, distributions) in enumerate(zip(ids, labels, qs)):
        for t in range(1, len(row)):
            for token, p in distributions[t].items():
                q[b, t-1, gen.token2id[token]] = p
            if labs[t] == 'mixed_verb':
                masks['verb'][b, t-1] = True
                candidates = gen.candidates(row[:t])
                if 'induction' in candidates:
                    masks['verb_induction'][b, t-1] = True
                else:
                    masks['verb_no_induction'][b, t-1] = True
                modal = {max(d, key=d.get) for d in candidates.values()}
                masks['conflict'][b, t-1] = len(modal) > 1
            elif labs[t] == 'frame':
                masks['frame'][b, t-1] = True
    assert np.allclose(q.sum(-1), 1)
    return dict(x=z[:, :-1], y=z[:, 1:], q=torch.from_numpy(q).to(device),
                masks={k: torch.from_numpy(v).to(device)
                       for k, v in masks.items()})


@torch.no_grad()
def evaluate(model, data):
    logits = torch.cat([model(batch)[0].float()
                        for batch in data['x'].split(32)])
    logp = F.log_softmax(logits, -1)
    q = data['q']
    entropy = -(q*q.clamp_min(1e-30).log()).sum(-1)
    kl = -(q*logp).sum(-1)-entropy
    return {name: dict(n=int(mask.sum()),
                       kl=float(kl[mask].mean()) if mask.any() else None,
                       oracle_entropy=float(entropy[mask].mean()) if mask.any() else None)
            for name, mask in data['masks'].items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--depths', default='2,3')
    p.add_argument('--steps', type=int, default=4000)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--length', type=int, default=71)
    p.add_argument('--width', type=int, default=64)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--lr', type=float, default=.003)
    p.add_argument('--verb-loss-weight', type=float, default=1.0,
                   help='Diagnostic upweighting of mixed-verb CE; 1 is ordinary LM')
    p.add_argument('--oracle-verb', action='store_true',
                   help='Use exact soft targets at verbs to remove sampling noise')
    p.add_argument('--longlag-extra-weight', type=float, default=1.0,
                   help='Further weight verb targets whose induction lag exceeds 32 tokens')
    p.add_argument('--neval-per-mode', type=int, default=256)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cpu')
    p.add_argument('--regime', choices=('full', 'pair'), default='full')
    p.add_argument('--output-dir', required=True)
    a = p.parse_args()
    assert a.verb_loss_weight >= 1 and a.longlag_extra_weight >= 1
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config

    torch.set_num_threads(4)
    device = torch.device(a.device)
    output = Path(a.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    test = {mode: make_eval(ConditionalSurfaceLanguage(
        seed=10101+i, mode_probs={m: float(m == mode) for m in MODES},
        regime=a.regime),
        a.neval_per_mode, a.length, device)
        for i, mode in enumerate(MODES)}
    records = []
    for depth in [int(x) for x in a.depths.split(',')]:
        torch.manual_seed(10200+a.seed)
        config = Config(vocab_size=len(VOCAB), n_layer=depth,
                        n_head=a.heads, n_embd=a.width, seq_len=a.length,
                        block_has_mlp=(False,)*depth, use_rmsnorm=True)
        model = BilinearGPT(config).to(device)
        gen = ConditionalSurfaceLanguage(seed=10300+a.seed, regime=a.regime)
        optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr)
        snapshots = []
        t0 = time.time()
        model.train()
        for step in range(a.steps):
            batch_data = make_batch(gen, a.batch, a.length, device,
                                    return_verb_mask=True,
                                    return_verb_q=a.oracle_verb,
                                    return_longlag=a.longlag_extra_weight > 1)
            x, y, verb = batch_data[:3]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)[0]
            token_loss = F.cross_entropy(logits.reshape(-1, len(VOCAB)),
                                         y.reshape(-1), reduction='none').reshape_as(y)
            if a.oracle_verb:
                q = batch_data[3]
                oracle_ce = -(q*F.log_softmax(logits, -1)).sum(-1)
                token_loss = torch.where(verb, oracle_ce, token_loss)
            weights = torch.where(verb, a.verb_loss_weight, 1.0)
            if a.longlag_extra_weight > 1:
                longlag = batch_data[-1]
                weights = torch.where(longlag,
                                      a.verb_loss_weight*a.longlag_extra_weight,
                                      weights)
            loss = (token_loss*weights).sum()/weights.sum()
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss at depth {depth} step {step}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            if (step+1) % 500 == 0 or step+1 == a.steps:
                model.eval()
                scores = {mode: evaluate(model, data)
                          for mode, data in test.items()}
                snapshots.append(dict(step=step+1, train_ce=float(loss.detach()),
                                      scores=scores, elapsed=time.time()-t0))
                print('depth', depth, 'step', step+1,
                      {mode: round(scores[mode]['verb_induction']['kl'], 4)
                       for mode in MODES},
                      'elapsed', round(time.time()-t0, 1), flush=True)
                model.train()
                (output/'results.json').write_text(json.dumps(dict(
                    config=vars(a), vocabulary=list(VOCAB), records=records,
                    current=dict(depth=depth, snapshots=snapshots)), indent=2)+'\n')
        torch.save(model.state_dict(), output/f'model_depth{depth}.pt')
        records.append(dict(depth=depth, model_config=asdict(config),
                            parameters=model.count_params(), snapshots=snapshots))
        (output/'results.json').write_text(json.dumps(dict(
            config=vars(a), vocabulary=list(VOCAB), records=records), indent=2)+'\n')


if __name__ == '__main__':
    main()

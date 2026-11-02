"""Same-vocabulary short-prefix alias lookup for curriculum transfer.

Uses the full readable alias vocabulary and exact verb oracle, but fixed short
supported prefixes. The saved model shape matches train_surface_mixture.py.
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from surface_alias_mixture import ALIASES, AliasMixtureLanguage
from surface_mixture_v2 import PEOPLE


WEIGHTS = dict(lexical=1., induction=.0001, lead=.0001,
               place_skip=.0001)
LENGTH = 20


def heldout(alias, person):
    return (alias+person) % 4 == 0


def cases(n, seed, split):
    if split not in ('train', 'pair_holdout'):
        raise ValueError(split)
    rng = np.random.default_rng(seed)
    rows = []
    while len(rows) < n:
        batch = max(256, 2*(n-len(rows)))
        a = rng.integers(len(ALIASES), size=(batch, 2))
        p = rng.integers(len(PEOPLE), size=(batch, 2))
        q = rng.integers(2, size=batch)
        for i in range(batch):
            a0, a1 = a[i]
            p0, p1 = p[i]
            if a0 == a1 or p0 == p1:
                continue
            h0, h1 = heldout(a0, p0), heldout(a1, p1)
            if split == 'train' and (h0 or h1):
                continue
            if split == 'pair_holdout' and (not (h0, h1)[q[i]]
                                            or (h0, h1)[1-q[i]]):
                continue
            rows.append((int(p0), int(a0), int(p1), int(a1), int(q[i])))
            if len(rows) == n:
                break
    return np.asarray(rows, dtype=np.int64)


def encode(gen, rows, edit='none'):
    p0, a0, p1, a1, q = rows.T
    if edit == 'order':
        p0, a0, p1, a1 = p1, a1, p0, a0
    elif edit == 'alias_labels':
        a0, a1 = a1, a0
    elif edit not in ('none', 'query'):
        raise ValueError(edit)
    query = 1-q if edit == 'query' else q
    alias = np.where(query == 0, rows[:, 1], rows[:, 3])
    target_person = np.where(alias == a0, p0, p1)
    token = gen.token2id
    person_ids = np.asarray([token[x] for x in PEOPLE])
    alias_ids = np.asarray([token[x] for x in ALIASES])
    x = np.empty((len(rows), LENGTH), dtype=np.int64)
    x[:, 0] = person_ids[p0]
    x[:, 1] = token['(']
    x[:, 2] = token['also']
    x[:, 3] = token['called']
    x[:, 4] = alias_ids[a0]
    x[:, 5] = token[')']
    x[:, 6] = token['.']
    x[:, 7] = person_ids[p1]
    x[:, 8] = token['(']
    x[:, 9] = token['also']
    x[:, 10] = token['called']
    x[:, 11] = alias_ids[a1]
    x[:, 12] = token[')']
    x[:, 13] = token['.']
    x[:, 14] = alias_ids[alias]
    x[:, 15] = token[',']
    x[:, 16] = token['before']
    x[:, 17] = token['the']
    x[:, 18] = token['storm']
    x[:, 19] = token[',']
    return x, target_person


def oracle_table(gen):
    # The fixed prompt has no prior narrative; its verb oracle depends only on
    # the canonical subject. Verify that this cache is valid on sampled cases.
    table = np.zeros((len(PEOPLE), len(gen.vocab)), dtype=np.float32)
    for person in range(len(PEOPLE)):
        other = (person+1) % len(PEOPLE)
        row = np.asarray([[person, 0, other, 1, 0]])
        x, _ = encode(gen, row)
        dist = gen.next_distribution(x[0].tolist())
        for token, prob in dist.items():
            table[person, gen.token2id[token]] = prob
    assert np.max(np.abs(table.sum(-1)-1)) < 1e-6
    probe = cases(100, 9021, 'train')
    for edit in ('none', 'query', 'order', 'alias_labels'):
        x, target = encode(gen, probe, edit)
        for prefix, person in zip(x, target):
            dist = gen.next_distribution(prefix.tolist())
            actual = np.zeros(len(gen.vocab))
            for token, prob in dist.items():
                actual[gen.token2id[token]] = prob
            assert np.max(np.abs(actual-table[person])) < 1e-6
    return table


@torch.no_grad()
def predict(model, x, device, batch_size=512):
    model.eval()
    out = []
    for start in range(0, len(x), batch_size):
        t = torch.tensor(x[start:start+batch_size], dtype=torch.long,
                         device=device)
        out.append(model(t)[0][:, -1].float().softmax(-1).cpu().numpy())
    return np.concatenate(out)


def evaluate(model, gen, table, rows, device):
    x, person = encode(gen, rows)
    p = predict(model, x, device)
    q = table[person]
    informative = np.zeros(len(rows), dtype=bool)
    x_switch, other = encode(gen, rows, 'query')
    p_switch = predict(model, x_switch, device)
    q_switch = table[other]
    oracle_delta = q_switch-q
    oracle_tv = np.abs(oracle_delta).sum(-1)/2
    informative[:] = oracle_tv > .1
    model_delta = p_switch-p
    x_order, order_person = encode(gen, rows, 'order')
    assert np.array_equal(person, order_person)
    p_order = predict(model, x_order, device)
    x_labels, label_person = encode(gen, rows, 'alias_labels')
    p_labels = predict(model, x_labels, device)
    label_delta = table[label_person]-q
    assert informative.sum() > len(rows)//2
    return dict(n=len(rows), informative_n=int(informative.sum()),
                oracle_kl=float(np.mean(np.sum(
                    q*(np.log(q.clip(1e-12))-np.log(p.clip(1e-12))), axis=-1))),
                query_oracle_tv=float(oracle_tv[informative].mean()),
                query_model_tv=float((np.abs(model_delta).sum(-1)/2)[informative].mean()),
                query_effect_l1=float(np.abs(model_delta-oracle_delta).sum(-1)[informative].mean()),
                order_model_tv=float(np.abs(p_order-p).sum(-1).mean()/2),
                label_oracle_tv=float(np.abs(label_delta).sum(-1)[informative].mean()/2),
                label_model_tv=float(np.abs(p_labels-p).sum(-1)[informative].mean()/2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--lr', type=float, default=.003)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config

    torch.set_num_threads(4)
    device = torch.device(a.device)
    gen = AliasMixtureLanguage(initial_alias_definitions=2,
                               parenthetical_rate=0,
                               definition_start_rate=0,
                               alias_use=.9,
                               alias_reuse_rate=.8,
                               weights=WEIGHTS)
    table = oracle_table(gen)
    torch.manual_seed(8200+a.seed)
    config = Config(vocab_size=len(gen.vocab), n_layer=2, n_head=4,
                    n_embd=64, seq_len=63, block_has_mlp=(False, False),
                    use_rmsnorm=True)
    model = BilinearGPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr)
    rng = np.random.default_rng(8300+a.seed)
    panels = dict(train_pair=cases(2048, 8400+a.seed, 'train'),
                  pair_holdout=cases(2048, 8500+a.seed, 'pair_holdout'))
    snapshots = []
    for step in range(1, a.steps+1):
        model.train()
        rows = cases(a.batch, int(rng.integers(2**31)), 'train')
        x, person = encode(gen, rows)
        xt = torch.tensor(x, dtype=torch.long, device=device)
        qt = torch.tensor(table[person], dtype=torch.float32, device=device)
        optimizer.zero_grad()
        logits = model(xt)[0][:, -1].float()
        loss = -(qt*F.log_softmax(logits, -1)).sum(-1).mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss at step {step}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
        optimizer.step()
        if step == 1 or step % 500 == 0 or step == a.steps:
            scores = {name: evaluate(model, gen, table, panel, device)
                      for name, panel in panels.items()}
            snapshots.append(dict(step=step, train_ce=float(loss.detach()),
                                  scores=scores))
            print(step, {name: (round(score['oracle_kl'], 4),
                                round(score['query_model_tv'], 4),
                                round(score['query_effect_l1'], 4))
                         for name, score in scores.items()}, flush=True)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), a.output_dir/'model_depth2.pt')
    (a.output_dir/'results.json').write_text(json.dumps(dict(
        config={**vars(a), 'output_dir': str(a.output_dir)},
        model_config=asdict(config), generator_kwargs=dict(
            initial_alias_definitions=2, parenthetical_rate=0,
            definition_start_rate=0, alias_use=.9, alias_reuse_rate=.8,
            weights=WEIGHTS), vocabulary=list(gen.vocab),
        split='(alias_index+person_index) % 4 == 0; both training definitions allowed',
        snapshots=snapshots), indent=2)+'\n')


if __name__ == '__main__':
    main()

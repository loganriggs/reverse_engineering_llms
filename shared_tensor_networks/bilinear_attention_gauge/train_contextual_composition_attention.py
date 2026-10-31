"""Train a query-conditioned bilinear attention toy with whole-family audit splits.

The model learns all tasks. Later featurizer fitting can exclude an entire
query family (XOR or AND) while still evaluating a competent native model.
"""

import argparse
import copy
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dual_joint_function_canonical import HERE
from train_dense_four_count_weak import SIGNS


N_KEYS = 7
PAIRS = tuple((a, b) for a in range(4) for b in range(a + 1, 4))
TASKS = tuple([f'bit_{i}' for i in range(4)] +
              [f'xor_{a}_{b}' for a, b in PAIRS] +
              [f'and_{a}_{b}' for a, b in PAIRS])
FAMILIES = {'primitive': tuple(range(4)),
            'xor': tuple(range(4, 10)),
            'and': tuple(range(10, 16))}


def targets(keys, query):
    """Exact labels from four independent token-bit majorities."""
    bits = SIGNS[keys].sum(axis=1) > 0
    all_targets = np.column_stack(
        [bits[:, i] for i in range(4)] +
        [bits[:, a] ^ bits[:, b] for a, b in PAIRS] +
        [bits[:, a] & bits[:, b] for a, b in PAIRS])
    return all_targets[np.arange(len(keys)), query].astype(np.float32)


def examples(n, seed, family=None, token_prob=None):
    rng = np.random.default_rng(seed)
    keys = rng.choice(16, size=(n, N_KEYS), p=token_prob)
    possible = np.arange(16) if family is None else np.array(FAMILIES[family])
    query = rng.choice(possible, size=n)
    x = np.column_stack((keys, 16 + query)).astype(np.int64)
    y = targets(keys, query)
    return torch.from_numpy(x), torch.from_numpy(y), query


class ContextualCompositionAttention(nn.Module):
    def __init__(self, width=32, heads=8, qk_width=4):
        super().__init__()
        self.width, self.heads, self.qk_width = width, heads, qk_width
        self.embed = nn.Embedding(32, width)
        for name in ('q1', 'k1', 'q2', 'k2'):
            setattr(self, name, nn.Linear(width, heads*qk_width, bias=False))
        self.v = nn.Linear(width, heads, bias=False)
        self.read = nn.Sequential(nn.Linear(heads + width, 64), nn.Tanh(),
                                  nn.Linear(64, 1))

    def head_outputs(self, x):
        z = self.embed(x)
        def projected(name):
            raw = getattr(self, name)(z).reshape(len(x), N_KEYS+1,
                                                  self.heads, self.qk_width)
            return F.rms_norm(raw, (self.qk_width,))
        q1, k1, q2, k2 = [projected(name) for name in
                          ('q1', 'k1', 'q2', 'k2')]
        s1 = torch.einsum('bhd,bshd->bsh', q1[:, -1], k1[:, :-1])/self.qk_width
        s2 = torch.einsum('bhd,bshd->bsh', q2[:, -1], k2[:, :-1])/self.qk_width
        values = self.v(z)[:, :-1]
        return (s1*s2*values).sum(1), z[:, -1]

    def forward(self, x):
        h, query = self.head_outputs(x)
        return self.read(torch.cat((h, query), dim=1)).squeeze(-1)


def evaluate(model, x, y, query):
    model.eval()
    with torch.no_grad():
        out = torch.cat([model(part) for part in x.split(4096)])
        loss = F.binary_cross_entropy_with_logits(out, y, reduction='none')
        correct = ((out > 0) == y.bool()).cpu().numpy()
    return {name: dict(n=int(np.isin(query, ids).sum()),
                       accuracy=float(correct[np.isin(query, ids)].mean()),
                       bce=float(loss[torch.from_numpy(np.isin(query, ids))].mean()))
            for name, ids in FAMILIES.items()}


def train(seed, steps=2500, ntrain=100000, nval=12000, ntest=30000):
    torch.set_num_threads(1)
    torch.manual_seed(20261130+seed)
    train_x, train_y, _ = examples(ntrain, 20261200+seed)
    val_x, val_y, val_query = examples(nval, 20261300+seed)
    test_x, test_y, test_query = examples(ntest, 20261400+seed)
    # A separate token-frequency shift is predeclared; its labels stay exact.
    prob = np.arange(1, 17, dtype=float)
    prob /= prob.sum()
    shift_x, shift_y, shift_query = examples(ntest, 20261500+seed,
                                             token_prob=prob)
    model = ContextualCompositionAttention()
    opt = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=1e-4)
    rng = np.random.default_rng(20261600+seed)
    best = (float('inf'), None, None)
    history = []
    for step in range(1, steps+1):
        model.train()
        idx = torch.from_numpy(rng.integers(0, ntrain, 512))
        loss = F.binary_cross_entropy_with_logits(model(train_x[idx]),
                                                  train_y[idx])
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.)
        opt.step()
        if step % 250 == 0 or step == 1 or step == steps:
            val = evaluate(model, val_x, val_y, val_query)
            test = evaluate(model, test_x, test_y, test_query)
            history.append(dict(step=step, train_bce=float(loss.detach()),
                                validation=val, test=test))
            score = sum(v['bce'] for v in val.values())/len(val)
            if score < best[0]:
                best = (score, copy.deepcopy(model.state_dict()), step)
            print(seed, step, 'val', {k: round(v['accuracy'], 3)
                                       for k, v in val.items()}, flush=True)
    model.load_state_dict(best[1])
    final = dict(validation=evaluate(model, val_x, val_y, val_query),
                 test=evaluate(model, test_x, test_y, test_query),
                 shifted=evaluate(model, shift_x, shift_y, shift_query))
    path = HERE/f'contextual_composition_attention_seed{seed}'
    torch.save(model.state_dict(), str(path)+'.pt')
    (HERE/f'{path.name}.json').write_text(json.dumps(dict(
        seed=seed, task_names=TASKS, families=FAMILIES,
        task_definition='four token-bit majorities, pairwise XOR and AND; query token selects one of 16 tasks',
        ntrain=ntrain, nval=nval, ntest=ntest,
        checkpoint_selection='mean validation BCE over three families',
        selected_step=best[2], history=history, selected=final), indent=2)+'\n')
    return final


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=2500)
    parser.add_argument('--ntrain', type=int, default=100000)
    parser.add_argument('--nval', type=int, default=12000)
    parser.add_argument('--ntest', type=int, default=30000)
    args = parser.parse_args()
    train(args.seed, args.steps, args.ntrain, args.nval, args.ntest)

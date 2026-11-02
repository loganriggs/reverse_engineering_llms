"""Two-binding capability control for the bilinear toy transformer.

The answer-only task is intentionally much simpler than the joint-rule stream:
P0 means A0 . P1 means A1 . ask Aq : -> Pq. Both aliases and people vary on
every example; a checkerboard of alias-person combinations is train-held-out.
"""

import argparse
import itertools
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from softmax_attention_control import configure_attention


NPERSON = 8
NALIAS = 8
MEANS, DOT, ASK, COLON = range(NPERSON+NALIAS, NPERSON+NALIAS+4)
VOCAB = tuple([f'person_{i}' for i in range(NPERSON)]
              + [f'alias_{i}' for i in range(NALIAS)]
              + ['means', '.', 'ask', ':'])
LENGTH = 11


def heldout(alias, person):
    return (alias+person) % 4 == 0


def sequence_holdout(c):
    """A fixed whole-sequence split among allowed alias-person pairs."""
    return ((13*c.alias0 + 17*c.person0 + 19*c.alias1
             + 23*c.person1 + 29*c.query) % 5 == 0)


@dataclass(frozen=True)
class Case:
    person0: int
    alias0: int
    person1: int
    alias1: int
    query: int


def all_cases(split):
    if split not in ('train', 'id_test', 'pair_holdout'):
        raise ValueError(split)
    cases = []
    for (a0, a1), (p0, p1), query in itertools.product(
            itertools.permutations(range(NALIAS), 2),
            itertools.permutations(range(NPERSON), 2), range(2)):
        c = Case(p0, a0, p1, a1, query)
        status = [heldout(a0, p0), heldout(a1, p1)]
        if split in ('train', 'id_test'):
            if any(status) or sequence_holdout(c) != (split == 'id_test'):
                continue
        if split == 'pair_holdout' and (not status[query]
                                        or status[1-query]):
            continue
        cases.append(c)
    return cases


def encode(c, edit='none'):
    pairs = [(c.person0, c.alias0), (c.person1, c.alias1)]
    if edit == 'order':
        pairs.reverse()
    elif edit == 'alias_labels':
        pairs = [(pairs[0][0], pairs[1][1]),
                 (pairs[1][0], pairs[0][1])]
    elif edit == 'people':
        pairs = [(pairs[1][0], pairs[0][1]),
                 (pairs[0][0], pairs[1][1])]
    elif edit not in ('none', 'query'):
        raise ValueError(edit)
    alias = (c.alias0, c.alias1)[1-c.query if edit == 'query' else c.query]
    x = [pairs[0][0], MEANS, NPERSON+pairs[0][1], DOT,
         pairs[1][0], MEANS, NPERSON+pairs[1][1], DOT,
         ASK, NPERSON+alias, COLON]
    binding = {a: p for p, a in pairs}
    return x, binding[alias]


def batch(cases, edit='none', device='cpu'):
    rows = [encode(c, edit) for c in cases]
    return (torch.tensor([x for x, _ in rows], dtype=torch.long, device=device),
            torch.tensor([y for _, y in rows], dtype=torch.long, device=device))


@torch.no_grad()
def predict(model, cases, edit, device, batch_size=512):
    model.eval()
    outputs = []
    targets = []
    for start in range(0, len(cases), batch_size):
        x, y = batch(cases[start:start+batch_size], edit, device)
        outputs.append(model(x)[0][:, -1].float().softmax(-1).cpu())
        targets.append(y.cpu())
    return torch.cat(outputs), torch.cat(targets)


def evaluate(model, cases, device):
    base, target = predict(model, cases, 'none', device)
    q = torch.zeros_like(base)
    q[torch.arange(len(target)), target] = 1
    result = dict(n=len(cases), accuracy=float((base.argmax(-1) == target).float().mean()),
                  answer_probability=float(base[torch.arange(len(target)), target].mean()),
                  answer_nll=float(-base[torch.arange(len(target)), target].clamp_min(1e-12).log().mean()))
    edits = {}
    for name in ('query', 'order', 'alias_labels', 'people'):
        changed, changed_target = predict(model, cases, name, device)
        oracle_changed = torch.zeros_like(base)
        oracle_changed[torch.arange(len(target)), changed_target] = 1
        assert bool((changed_target == target).all()) == (name == 'order')
        model_effect = changed-base
        oracle_effect = oracle_changed-q
        edits[name] = dict(
            oracle_tv=float(oracle_effect.abs().sum(-1).mean()/2),
            model_tv=float(model_effect.abs().sum(-1).mean()/2),
            effect_l1_error=float((model_effect-oracle_effect).abs().sum(-1).mean()),
            edited_accuracy=float((changed.argmax(-1) == changed_target).float().mean()))
    result['edits'] = edits
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--attention', choices=('bilinear', 'softmax'), default='bilinear')
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--width', type=int, default=64)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--block-mlp', action='store_true',
                   help='Add the model block MLP as a capability control')
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--lr', type=float, default=.003)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cpu')
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config

    torch.set_num_threads(4)
    device = torch.device(a.device)
    torch.manual_seed(7300+a.seed)
    config = Config(vocab_size=len(VOCAB), n_layer=a.depth, n_head=a.heads,
                    n_embd=a.width, seq_len=LENGTH,
                    block_has_mlp=(a.block_mlp,)*a.depth, use_rmsnorm=True)
    model = configure_attention(BilinearGPT(config), a.attention).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr)
    train_cases = all_cases('train')
    rng = np.random.default_rng(7500+a.seed)
    panels = dict(id_test=all_cases('id_test'),
                  pair_holdout=all_cases('pair_holdout'))
    assert all(not heldout(a, p) for c in train_cases
               for a, p in ((c.alias0, c.person0), (c.alias1, c.person1)))
    assert all(not sequence_holdout(c) for c in train_cases)
    assert all(sequence_holdout(c) for c in panels['id_test'])
    train_inputs = {tuple(encode(c)[0]) for c in train_cases}
    assert not any(tuple(encode(c)[0]) in train_inputs
                   for c in panels['id_test']+panels['pair_holdout'])
    assert all(heldout((c.alias0, c.alias1)[c.query],
                       (c.person0, c.person1)[c.query])
               for c in panels['pair_holdout'])
    snapshots = []
    for step in range(1, a.steps+1):
        model.train()
        selected = rng.integers(len(train_cases), size=a.batch)
        x, y = batch([train_cases[i] for i in selected], device=device)
        optimizer.zero_grad()
        logits = model(x)[0][:, -1].float()
        loss = F.cross_entropy(logits, y)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss at step {step}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        if step == 1 or step % 500 == 0 or step == a.steps:
            scores = {name: evaluate(model, cases, device)
                      for name, cases in panels.items()}
            snapshots.append(dict(step=step, train_loss=float(loss.detach()),
                                  scores=scores))
            print(step, {name: {key: round(row[key], 4)
                                for key in ('accuracy', 'answer_nll')}
                         for name, row in scores.items()}, flush=True)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), a.output_dir/'model.pt')
    result = dict(config={**vars(a), 'output_dir': str(a.output_dir)},
                  model_config=asdict(config), vocab=VOCAB,
                  heldout_rule='(alias_index+person_index) % 4 == 0',
                  sequence_holdout_rule='(13*a0+17*p0+19*a1+23*p1+29*query) % 5 == 0',
                  train_cases=len(train_cases), snapshots=snapshots)
    (a.output_dir/'results.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()

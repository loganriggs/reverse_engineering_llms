"""Train one bilinear transformer conditioned on a rule-mixture token.

Generated tokens still follow the original Python LanguageGenerator. The
mixture tag is external conditioning information, not part of its rule context.
"""

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from train_mixed_rule_weight_sweep import MIXTURES, make_eval, evaluate


NAMES = ('baseline', 'bigram_heavy', 'induction_heavy')


def tagged_batch(gen, name, n, length, device):
    ids, _ = gen.sample_batch(n, length)
    z = torch.tensor(ids, dtype=torch.long, device=device)
    tag = torch.full((n, 1), 26+NAMES.index(name), dtype=torch.long, device=device)
    return torch.cat([tag, z[:, :-1]], 1), z


def tagged_eval(gen, name, n, length, device):
    item = make_eval(gen, n, length, device)
    x, y = item['x'], item['y']
    tag = torch.full((n, 1), 26+NAMES.index(name), dtype=torch.long, device=device)
    first = x[:, :1]
    item['x'] = torch.cat([tag, x], 1)
    item['y'] = torch.cat([first, y], 1)
    first_q = torch.zeros((n, 1, 29), device=device)
    first_q[:, :, :26] = 1/26
    q = torch.zeros((n, length-1, 29), device=device)
    q[:, :, :26] = item['q']
    item['q'] = torch.cat([first_q, q], 1)
    item['conflict'] = torch.cat([torch.zeros((n, 1), dtype=torch.bool, device=device),
                                   item['conflict']], 1)
    import numpy as np
    item['pair_types'] = np.concatenate([np.full((n, 1), '', dtype=object),
                                          item['pair_types']], 1)
    return item


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--steps', type=int, default=10000)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--length', type=int, default=64)
    p.add_argument('--width', type=int, default=64)
    p.add_argument('--n-layer', type=int, default=2)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--lr', type=float, default=.003)
    p.add_argument('--neval', type=int, default=256)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cpu')
    p.add_argument('--output-dir', required=True)
    a = p.parse_args()
    sys.path.insert(0, a.source)
    from generator import LanguageGenerator, ALL_RULES
    from model import BilinearGPT, Config

    torch.set_num_threads(4)
    device = torch.device(a.device)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = Config(vocab_size=29, n_layer=a.n_layer, n_head=a.heads,
                    n_embd=a.width, seq_len=a.length,
                    block_has_mlp=(False,)*a.n_layer)
    tests = {name: tagged_eval(LanguageGenerator(ALL_RULES,
                    mixing_weights=MIXTURES[name], mode='mixed', seed=9500+i),
                    name, a.neval, a.length, device)
             for i, name in enumerate(NAMES)}
    train_gens = {name: LanguageGenerator(ALL_RULES, mixing_weights=MIXTURES[name],
                                         mode='mixed', seed=9700+i+a.seed*100)
                  for i, name in enumerate(NAMES)}
    torch.manual_seed(9900+a.seed)
    model = BilinearGPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr)
    t0 = time.time()
    checkpoints = []
    for step in range(a.steps):
        name = NAMES[step % len(NAMES)]
        x, y = tagged_batch(train_gens[name], name, a.batch, a.length, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)[0]
        loss = F.cross_entropy(logits.reshape(-1, 29), y.reshape(-1))
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss at step {step}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        if (step+1) % 2000 == 0 or step+1 == a.steps:
            model.eval()
            result = {name: evaluate(model, item) for name, item in tests.items()}
            checkpoints.append(dict(step=step+1, train_ce=float(loss.detach()),
                                    seconds=time.time()-t0, evaluations=result))
            print('step', step+1, 'train CE', round(float(loss.detach()), 4),
                  'conflict KL', {name: round(result[name]['conflict']['kl'], 4)
                                  for name in NAMES},
                  'elapsed', round(time.time()-t0, 1), flush=True)
            model.train()
    torch.save(model.state_dict(), out/'model.pt')
    (out/'model_config.json').write_text(json.dumps(asdict(config), indent=2)+'\n')
    (out/'results.json').write_text(json.dumps(dict(config=vars(a),
            model_config=asdict(config), mixtures=MIXTURES, checkpoints=checkpoints),
            indent=2)+'\n')


if __name__ == '__main__':
    main()

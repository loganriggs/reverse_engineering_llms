"""Train small bilinear language models on controlled rule-mixture interventions.

Uses the exact Python LanguageGenerator rather than the fast approximation.
Every trained model is evaluated on every held-out mixture with observable
oracle KL, so changes in entropy are not mistaken for changes in competence.
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

from mixed_rule_oracle import observable_distributions_for_sequence


SOURCE = '/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits'
MIXTURES = {
    'baseline': {'bigram': .25, 'trigram': .22, 'skip_bigram': .22,
                 'induction': .20, 'bracket': .01, 'noise': .10},
    'bigram_heavy': {'bigram': .35, 'trigram': .22, 'skip_bigram': .22,
                     'induction': .10, 'bracket': .01, 'noise': .10},
    'induction_heavy': {'bigram': .10, 'trigram': .22, 'skip_bigram': .22,
                       'induction': .35, 'bracket': .01, 'noise': .10},
}


def tensor_batch(gen, n, length, device):
    ids, _ = gen.sample_batch(n, length)
    z = torch.tensor(ids, dtype=torch.long, device=device)
    return z[:, :-1], z[:, 1:], ids


def make_eval(gen, n, length, device):
    x, y, rows = tensor_batch(gen, n, length, device)
    q = np.zeros((n, length-1, len(gen.vocab)), dtype=np.float32)
    conflict = np.zeros((n, length-1), dtype=bool)
    pair_types = np.empty((n, length-1), dtype=object)
    for b, ids in enumerate(rows):
        tokens = [gen.id2token[i] for i in ids]
        distributions = observable_distributions_for_sequence(gen, tokens)
        for t in range(1, length):
            for token, mass in distributions[t].items():
                q[b, t-1, gen.token2id[token]] = mass
            candidates = []
            for cls, fn in [('bigram', gen._try_bigram),
                            ('trigram', gen._try_trigram),
                            ('skip_bigram', gen._try_skip_bigram),
                            ('induction', gen._try_induction)]:
                token, _ = fn(tokens[:t][-gen.window_size:])
                if token is not None:
                    candidates.append((cls, token))
            conflict[b, t-1] = len({tok for _, tok in candidates}) > 1
            pair_types[b, t-1] = '+'.join(sorted(cls for cls, _ in candidates))
    return dict(x=x, y=y, q=torch.from_numpy(q).to(device),
                conflict=torch.from_numpy(conflict).to(device),
                pair_types=pair_types)


@torch.no_grad()
def evaluate(model, item):
    logits = torch.cat([model(batch)[0].float() for batch in item['x'].split(32)])
    logp = F.log_softmax(logits, -1)
    q = item['q']
    entropy = -(q*q.clamp_min(1e-30).log()).sum(-1)
    kl = -(q*logp).sum(-1)-entropy
    observed_ce = -logp.gather(-1, item['y'].unsqueeze(-1)).squeeze(-1)
    out = {}
    for name, mask in [('all', torch.ones_like(item['conflict'])),
                       ('conflict', item['conflict']),
                       ('no_conflict', ~item['conflict'])]:
        out[name] = dict(n=int(mask.sum()), kl=float(kl[mask].mean()),
                         oracle_entropy=float(entropy[mask].mean()),
                         observed_ce=float(observed_ce[mask].mean()))
    for pair in np.unique(item['pair_types']):
        if '+' not in pair:
            continue
        mask = torch.from_numpy(item['pair_types'] == pair).to(kl.device) & item['conflict']
        if int(mask.sum()) >= 20:
            out[pair] = dict(n=int(mask.sum()), kl=float(kl[mask].mean()),
                             oracle_entropy=float(entropy[mask].mean()))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default=SOURCE)
    p.add_argument('--mixtures', default='baseline,bigram_heavy,induction_heavy')
    p.add_argument('--seeds', type=int, default=2)
    p.add_argument('--steps', type=int, default=2500)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--length', type=int, default=64)
    p.add_argument('--n-layer', type=int, default=2)
    p.add_argument('--width', type=int, default=32)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--lr', type=float, default=.003)
    p.add_argument('--neval', type=int, default=128)
    p.add_argument('--device', default='cpu')
    p.add_argument('--eval-only', action='store_true',
                   help='Re-evaluate the saved checkpoints without retraining')
    p.add_argument('--output-dir', required=True)
    a = p.parse_args()
    sys.path.insert(0, a.source)
    from generator import LanguageGenerator, ALL_RULES
    from model import BilinearGPT, Config

    names = a.mixtures.split(',')
    assert all(name in MIXTURES for name in names)
    torch.set_num_threads(4)
    device = torch.device(a.device)
    output_dir = Path(a.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = Config(vocab_size=26, n_layer=a.n_layer, n_head=a.heads,
                    n_embd=a.width, seq_len=a.length-1,
                    block_has_mlp=(False,)*a.n_layer)
    evaluation = {}
    for i, name in enumerate(names):
        gen = LanguageGenerator(ALL_RULES, mixing_weights=MIXTURES[name],
                                mode='mixed', seed=9300+i)
        evaluation[name] = make_eval(gen, a.neval, a.length, device)
    results_path = output_dir / 'results.json'
    if a.eval_only:
        saved = json.loads(results_path.read_text())
        assert saved['config']['length'] == a.length
        assert saved['config']['neval'] == a.neval
        assert saved['config']['width'] == a.width
        assert saved['config']['n_layer'] == a.n_layer
        assert saved['config']['mixtures'] == a.mixtures
        for record in saved['records']:
            model = BilinearGPT(config).to(device)
            path = output_dir / f"model_{record['train_mixture']}_seed{record['seed']}.pt"
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            model.eval()
            record['evaluations'] = {test: evaluate(model, item)
                                     for test, item in evaluation.items()}
        results_path.write_text(json.dumps(saved, indent=2)+'\n')
        print('Re-evaluated', len(saved['records']), 'models')
        return
    records = []
    for seed in range(a.seeds):
        for i, name in enumerate(names):
            torch.manual_seed(6000+seed)
            model = BilinearGPT(config).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr)
            gen = LanguageGenerator(ALL_RULES, mixing_weights=MIXTURES[name],
                                    mode='mixed', seed=7000+seed)
            t0 = time.time()
            model.train()
            for step in range(a.steps):
                x, y, _ = tensor_batch(gen, a.batch, a.length, device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)[0]
                loss = F.cross_entropy(logits.reshape(-1, 26), y.reshape(-1))
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Non-finite loss at {name}, {seed}, {step}')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
                if (step+1) % 500 == 0:
                    print(name, seed, step+1, 'train CE', round(float(loss.detach()), 4),
                  'elapsed', round(time.time()-t0, 1), flush=True)
            model.eval()
            result = dict(train_mixture=name, seed=seed, steps=a.steps,
                          final_train_ce=float(loss.detach()), seconds=time.time()-t0,
                          evaluations={test: evaluate(model, item)
                                       for test, item in evaluation.items()})
            records.append(result)
            path = output_dir / f'model_{name}_seed{seed}.pt'
            torch.save(model.state_dict(), path)
            results_path.write_text(json.dumps(dict(
                config=vars(a), model_config=asdict(config), mixtures=MIXTURES,
                records=records), indent=2)+'\n')
            print('RESULT', name, seed,
                  {test: round(result['evaluations'][test]['conflict']['kl'], 4)
                   for test in names}, flush=True)


if __name__ == '__main__':
    main()

"""Compare learned bigram-vs-induction log-odds shifts on identical prefixes."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mixed_rule_oracle import observable_distributions_for_sequence
from train_mixed_rule_weight_sweep import MIXTURES


def metrics(target, predicted):
    target = np.asarray(target)
    predicted = np.asarray(predicted)
    return dict(n=len(target), target_mean=float(target.mean()),
                target_std=float(target.std()),
                predicted_mean=float(predicted.mean()),
                predicted_std=float(predicted.std()),
                mae=float(np.abs(target-predicted).mean()),
                rmse=float(np.sqrt(np.mean((target-predicted)**2))),
                constant_mean_rmse=float(target.std()),
                correlation=float(np.corrcoef(target, predicted)[0, 1]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--sweep-dir', default='language_task_suite/mixed_rule_weight_sweep_v1')
    p.add_argument('--conditional-dir', default=None,
                   help='Evaluate one model under each mixture tag instead of separately trained models')
    p.add_argument('--n', type=int, default=256)
    p.add_argument('--length', type=int, default=64)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    sys.path.insert(0, a.source)
    from generator import LanguageGenerator, ALL_RULES
    from model import BilinearGPT, Config

    torch.set_num_threads(4)
    names = ('baseline', 'bigram_heavy', 'induction_heavy')
    gens = {name: LanguageGenerator(ALL_RULES, mixing_weights=MIXTURES[name],
                                    mode='mixed', seed=9400)
            for name in names}
    ids, _ = gens['baseline'].sample_batch(a.n, a.length)
    tokens = [[gens['baseline'].id2token[i] for i in row] for row in ids]
    x = torch.tensor([row[:-1] for row in ids], dtype=torch.long)
    positions = []
    for b, row in enumerate(tokens):
        for t in range(1, a.length):
            context = row[:t]
            bigram, _ = gens['baseline']._try_bigram(context)
            induction, _ = gens['baseline']._try_induction(context)
            trigram, _ = gens['baseline']._try_trigram(context)
            skip, _ = gens['baseline']._try_skip_bigram(context)
            if (bigram is not None and induction is not None and
                    bigram != induction and trigram is None and skip is None):
                positions.append((b, t-1, gens['baseline'].token2id[bigram],
                                  gens['baseline'].token2id[induction]))
    assert positions
    oracle = {}
    for name, gen in gens.items():
        odds = []
        for b, row in enumerate(tokens):
            distributions = observable_distributions_for_sequence(gen, row)
            for bb, t, bigram_id, induction_id in positions:
                if bb != b:
                    continue
                bigram = gen.id2token[bigram_id]
                induction = gen.id2token[induction_id]
                q = distributions[t+1]
                numer, denom = q.get(bigram, 0.0), q.get(induction, 0.0)
                odds.append(np.log(numer/denom) if numer > 0 and denom > 0
                            else np.nan)
        oracle[name] = np.asarray(odds)
    supported = np.logical_and.reduce([np.isfinite(v) for v in oracle.values()])
    positions = [pos for pos, keep in zip(positions, supported) if keep]
    oracle = {name: values[supported] for name, values in oracle.items()}
    preds = {}
    if a.conditional_dir:
        conditional_dir = Path(a.conditional_dir)
        saved = json.loads((conditional_dir/'results.json').read_text())
        assert saved['config']['length'] == a.length
        config = Config(**saved['model_config'])
        model = BilinearGPT(config)
        model.load_state_dict(torch.load(conditional_dir/'model.pt', weights_only=True,
                                         map_location='cpu'))
        model.eval()
        for i, name in enumerate(names):
            tag = torch.full((len(x), 1), 26+i, dtype=torch.long)
            tagged_x = torch.cat([tag, x], 1)
            with torch.no_grad():
                logp = torch.cat([F.log_softmax(model(batch)[0], -1)
                                  for batch in tagged_x.split(32)])
            preds[(name, 0)] = np.asarray([
                float(logp[b, t+1, bigram_id]-logp[b, t+1, induction_id])
                for b, t, bigram_id, induction_id in positions])
        seeds = 1
    else:
        sweep = json.loads((Path(a.sweep_dir)/'results.json').read_text())
        config = Config(**sweep['model_config'])
        for record in sweep['records']:
            name, seed = record['train_mixture'], record['seed']
            model = BilinearGPT(config)
            path = Path(a.sweep_dir)/f'model_{name}_seed{seed}.pt'
            model.load_state_dict(torch.load(path, weights_only=True, map_location='cpu'))
            model.eval()
            with torch.no_grad():
                logp = torch.cat([F.log_softmax(model(batch)[0], -1)
                                  for batch in x.split(32)])
            preds[(name, seed)] = np.asarray([
                float(logp[b, t, bigram_id]-logp[b, t, induction_id])
                for b, t, bigram_id, induction_id in positions])
        seeds = sweep['config']['seeds']
    comparisons = {}
    for start, end in [('baseline', 'bigram_heavy'),
                       ('baseline', 'induction_heavy'),
                       ('bigram_heavy', 'induction_heavy')]:
        target = oracle[end]-oracle[start]
        for seed in range(seeds):
            predicted = preds[(end, seed)]-preds[(start, seed)]
            comparisons[f'{start}->{end}:seed{seed}'] = metrics(target, predicted)
    output = dict(config=vars(a), n_fixed_bigram_induction_conflicts=len(positions),
                  target_log_odds={name: dict(mean=float(v.mean()), std=float(v.std()))
                                   for name, v in oracle.items()},
                  model_log_odds={f'{name}:seed{seed}':
                                  dict(mean=float(v.mean()), std=float(v.std()))
                                  for (name, seed), v in preds.items()},
                  comparisons=comparisons,
                  caveat=('Same baseline-generated visible prefixes for every mixture. '
                          'The conditional run changes only the mixture tag in one model; '
                          'separately trained models are not parameter-aligned. '
                          'This tests behavioral adaptation, not circuit identity.'))
    Path(a.output).write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(comparisons, indent=2))


if __name__ == '__main__':
    main()

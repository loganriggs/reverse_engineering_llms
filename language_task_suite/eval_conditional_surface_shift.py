"""Fixed-prefix rule-weight interventions on a readable joint-rule DGP."""

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from conditional_surface_language import (MODES, ConditionalSurfaceLanguage)
from eval_surface_mixture_edits import predict
from surface_mixture_language import _history_and_clause


def make_pairs(n, length, seed, prefix_source='Balanced', regime='full',
               frame_stride=11):
    source_probs = {mode: float(mode == prefix_source) for mode in MODES}
    sampler = ConditionalSurfaceLanguage(seed=seed, mode_probs=source_probs,
                                         regime=regime)
    gen = ConditionalSurfaceLanguage(seed=seed+1, regime=regime)
    rows, labels, _ = sampler.sample_batch(n, length+1)
    pairs = []
    for row, labs in zip(rows, labels):
        for t in range(2, length+1):
            label = labs[t]
            if label == 'frame' and t % frame_stride:
                continue
            if label not in ('frame', 'mixed_verb'):
                continue
            prefix = ['Balanced']+[gen.id2token[i] for i in row[1:t]]
            candidates = gen.candidates(prefix) if label == 'mixed_verb' else {}
            induction = 'induction' in candidates
            induction_lag = None
            if induction:
                history, clause = _history_and_clause(prefix[2:])
                matches = [i for i, old in enumerate(history)
                           if old[2] == clause[2]]
                prior_verb = sum(len(old) for old in history[:matches[-1]])+3
                induction_lag = len(prefix[2:])-prior_verb
            lexical_token = (max(candidates['lexical'],
                                 key=candidates['lexical'].get)
                             if candidates else None)
            induction_token = (next(iter(candidates['induction']))
                               if induction else None)
            conflict = induction and lexical_token != induction_token
            for mode in MODES[1:]:
                changed = [mode]+prefix[1:]
                q1, _ = gen.next_distribution(changed)
                q0, _ = gen.next_distribution(prefix)
                if label == 'frame':
                    assert q0 == q1
                pairs.append(dict(kind=('frame' if label == 'frame' else
                                        'induction_conflict' if conflict else
                                        'other_verb'),
                                  mode=mode,
                                  lexical=lexical_token,
                                  induction=induction_token,
                                  induction_lag=induction_lag,
                                  target_position=t,
                                  before=[gen.token2id[x] for x in prefix],
                                  after=[gen.token2id[x] for x in changed],
                                  q0=q0, q1=q1))
    return gen, pairs


def score(pairs, p0, p1, gen):
    groups = defaultdict(list)
    for i, pair in enumerate(pairs):
        q0 = np.zeros(len(gen.vocab))
        q1 = np.zeros(len(gen.vocab))
        for token, mass in pair['q0'].items():
            q0[gen.token2id[token]] = mass
        for token, mass in pair['q1'].items():
            q1[gen.token2id[token]] = mass
        oracle_effect = q1-q0
        model_effect = p1[i]-p0[i]
        record = dict(oracle_tv=.5*np.abs(oracle_effect).sum(),
                      model_tv=.5*np.abs(model_effect).sum(),
                      effect_l1_error=np.abs(model_effect-oracle_effect).sum(),
                      induction_lag=pair['induction_lag'])
        if pair['kind'] == 'induction_conflict':
            lex = gen.token2id[pair['lexical']]
            ind = gen.token2id[pair['induction']]
            oracle_shift = math.log(q1[lex]/q1[ind])-math.log(q0[lex]/q0[ind])
            model_shift = (math.log(max(float(p1[i, lex]), 1e-30)/
                                    max(float(p1[i, ind]), 1e-30))-
                           math.log(max(float(p0[i, lex]), 1e-30)/
                                    max(float(p0[i, ind]), 1e-30)))
            record['oracle_log_odds_shift'] = oracle_shift
            record['model_log_odds_shift'] = model_shift
        groups[(pair['mode'], pair['kind'])].append(record)
    out = {}
    for (mode, kind), rows in groups.items():
        result = dict(n=len(rows))
        for field in ('oracle_tv', 'model_tv', 'effect_l1_error'):
            result[field] = float(np.mean([r[field] for r in rows]))
        if kind == 'induction_conflict':
            true = np.array([r['oracle_log_odds_shift'] for r in rows])
            pred = np.array([r['model_log_odds_shift'] for r in rows])
            result.update(oracle_mean_log_odds_shift=float(true.mean()),
                          model_mean_log_odds_shift=float(pred.mean()),
                          oracle_std_log_odds_shift=float(true.std()),
                          model_std_log_odds_shift=float(pred.std()),
                          shift_correlation=float(np.corrcoef(true, pred)[0, 1])
                          if true.std() > 1e-12 and pred.std() > 1e-12 else None,
                          log_odds_shift_rmse=float(np.sqrt(np.mean((pred-true)**2))),
                          constant_oracle_mean_rmse=float(np.sqrt(np.mean(
                              (true-true.mean())**2))))
            lag_groups = defaultdict(list)
            for row in rows:
                lag = row['induction_lag']
                key = '1-16' if lag <= 16 else '17-32' if lag <= 32 else '33+'
                lag_groups[key].append(row)
            result['by_induction_lag'] = {
                key: dict(n=len(subset),
                          oracle_mean=float(np.mean([
                              r['oracle_log_odds_shift'] for r in subset])),
                          model_mean=float(np.mean([
                              r['model_log_odds_shift'] for r in subset])),
                          rmse=float(np.sqrt(np.mean([
                              (r['oracle_log_odds_shift']-
                               r['model_log_odds_shift'])**2 for r in subset]))))
                for key, subset in lag_groups.items()}
        out.setdefault(mode, {})[kind] = result
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dir', required=True)
    p.add_argument('--n', type=int, default=512)
    p.add_argument('--length', type=int, default=71)
    p.add_argument('--seed', type=int, default=10401)
    p.add_argument('--prefix-source', choices=MODES, default='Balanced')
    p.add_argument('--regime', choices=('full', 'pair'), default='full')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config

    torch.set_num_threads(4)
    gen, pairs = make_pairs(a.n, a.length, a.seed, a.prefix_source, a.regime)
    before = [p['before'] for p in pairs]
    after = [p['after'] for p in pairs]
    saved = json.loads((Path(a.model_dir)/'results.json').read_text())
    assert saved['config'].get('regime', 'full') == a.regime, (
        'Model and evaluation regimes differ')
    results = {}
    for record in saved['records']:
        depth = record['depth']
        model = BilinearGPT(Config(**record['model_config']))
        model.load_state_dict(torch.load(Path(a.model_dir)/f'model_depth{depth}.pt',
                                         weights_only=True, map_location='cpu'))
        model.eval()
        p0 = predict(model, before, len(gen.vocab))
        p1 = predict(model, after, len(gen.vocab))
        results[str(depth)] = score(pairs, p0, p1, gen)
        print('depth', depth,
              {mode: round(results[str(depth)][mode]['induction_conflict'][
                  'log_odds_shift_rmse'], 3) for mode in MODES[1:]}, flush=True)
    output = dict(config=vars(a), n_pairs=len(pairs),
                  results=results,
                  caveat='A fixed body prefix is valid under every mode because '
                         'all rule weights are positive. The no-effect frame '
                         'control tests unintended mode dependence. This is '
                         'one trained model per depth and one sampled set of '
                         'fixed prefixes; a constant-shift baseline is included.')
    output['kinds'] = {f'{mode}:{kind}': count
                       for (mode, kind), count in Counter(
                           (pair['mode'], pair['kind']) for pair in pairs).items()}
    Path(a.output).write_text(json.dumps(output, indent=2)+'\n')


if __name__ == '__main__':
    main()

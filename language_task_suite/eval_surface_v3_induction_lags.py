"""Stratify previous-object causal effects by retrieval distance."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from eval_surface_mixture_edits import predict
from eval_surface_mixture_v3_edits import build_object_pairs
from surface_mixture_v3 import SurfaceMixtureV3


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dirs', nargs='+', required=True)
    p.add_argument('--n', type=int, default=1024)
    p.add_argument('--length', type=int, default=72)
    p.add_argument('--seed', type=int, default=8701)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config

    gen = SurfaceMixtureV3(seed=a.seed)
    pairs = [p for p in build_object_pairs(gen, a.n, a.length)
             if p['kind'] == 'object_induction']
    lags = []
    oracle = []
    for pair in pairs:
        index = next(i for i, (x, y) in enumerate(zip(pair['original'],
                                                       pair['edited'])) if x != y)
        lags.append(len(pair['original'])-index)
        q0 = np.zeros(len(gen.vocab), dtype=np.float32)
        q1 = q0.copy()
        for token, mass in pair['q_before'].items():
            q0[gen.token2id[token]] = mass
        for token, mass in pair['q_after'].items():
            q1[gen.token2id[token]] = mass
        oracle.append(q1-q0)
    oracle = np.stack(oracle)
    results = {}
    for directory in a.model_dirs:
        folder = Path(directory)
        saved = json.loads((folder/'results.json').read_text())
        record = saved['records'][0]
        model = BilinearGPT(Config(**record['model_config']))
        model.load_state_dict(torch.load(folder/f'model_depth{record["depth"]}.pt',
                                         map_location='cpu', weights_only=True))
        model.eval()
        p0 = predict(model, [p['original'] for p in pairs], len(gen.vocab))
        p1 = predict(model, [p['edited'] for p in pairs], len(gen.vocab))
        effect = p1-p0
        grouped = defaultdict(list)
        for i, lag in enumerate(lags):
            group = ('short_1_24' if lag <= 24 else
                     'medium_25_48' if lag <= 48 else 'long_49_plus')
            grouped[group].append(i)
        results[folder.name] = {
            name: dict(n=len(indices),
                       mean_lag=float(np.mean(np.array(lags)[indices])),
                       effect_l1_error=float(np.abs(effect[indices]-oracle[indices]).sum(-1).mean()),
                       model_effect_tv=float(np.abs(effect[indices]).sum(-1).mean()/2),
                       oracle_effect_tv=float(np.abs(oracle[indices]).sum(-1).mean()/2))
            for name, indices in grouped.items()}
    output = dict(config=vars(a), n_pairs=len(pairs), results=results,
                  caveat='Lag is measured from the changed prior object token '
                         'to the current object prediction. Long-lag strata '
                         'may contain too few pairs for a stable estimate.')
    Path(a.output).write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()

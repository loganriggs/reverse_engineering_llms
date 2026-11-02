"""Measure whether high first-layer token-pair scores occur in generated text."""

import argparse
import json
import sys
from pathlib import Path

import torch

from fit_varied_functional_sparsity import (functional_score_tables,
                                            observed_pair_weights)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dirs', nargs='+', type=Path, required=True)
    p.add_argument('--streams', type=int, default=8192)
    p.add_argument('--length', type=int, default=72)
    p.add_argument('--seed', type=int, default=20261202)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config

    settings = json.loads((a.model_dirs[0]/'results.json').read_text())
    kwargs = settings['generator_kwargs']
    lags = (1, 4, 16)
    weights, counts = observed_pair_weights(
        kwargs, lags, a.streams, a.length, a.seed,
        settings['records'][0]['model_config']['vocab_size'])
    results = {}
    for directory in a.model_dirs:
        metadata = json.loads((directory/'results.json').read_text())
        if metadata['generator_kwargs'] != kwargs:
            raise ValueError(f'Generator mismatch: {directory}')
        model = BilinearGPT(Config(**metadata['records'][0]['model_config']))
        model.load_state_dict(torch.load(directory/'model_depth2.pt',
                                         weights_only=True, map_location='cpu'))
        model.eval()
        rows = {}
        with torch.no_grad():
            for lag, table, weight in zip(lags, functional_score_tables(model),
                                          weights):
                values = table.abs().flatten()
                n_top = max(1, int(.01*values.numel()))
                top_values, top_indices = values.topk(n_top)
                support = (weight > 0).expand(table.shape[0], -1, -1).flatten()
                rows[str(lag)] = dict(
                    top_one_percent_abs_mass=float(top_values.sum()/values.sum()),
                    top_one_percent_entries_seen_fraction=float(
                        support[top_indices].float().mean()),
                    top_one_percent_mass_seen_fraction=float(
                        top_values[support[top_indices]].sum()/top_values.sum()),
                    all_abs_mass_seen_fraction=float(values[support].sum()/
                                                     values.sum()))
        results[str(directory)] = rows
    result = dict(config={**vars(a), 'model_dirs': [str(x) for x in a.model_dirs],
                          'output': str(a.output)},
                  observed_pair_counts=dict(zip(map(str, lags), counts)),
                  results=results,
                  caveat='Seen means present in a finite held-out generator '
                         'sample, not logically permitted by the grammar. '
                         'The same support is used for all models.')
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['results'], indent=2))


if __name__ == '__main__':
    main()

"""Describe token loadings and lag continuity of a weight-only score mode."""

import argparse
import json
import sys
from pathlib import Path

import torch

from eval_multi_genre_token_tensor import score_tables
from multi_genre_language import MultiGenreLanguage


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--depth', type=int, default=3)
    p.add_argument('--head', type=int, required=True)
    p.add_argument('--index', type=int, default=1)
    p.add_argument('--length', type=int, default=71)
    p.add_argument('--lags', default='0,1,4,8,16,32,64')
    p.add_argument('--top', type=int, default=10)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config
    torch.set_num_threads(4)
    saved = json.loads((a.model_dir/'results.json').read_text())
    record = next(row for row in saved['records'] if row['depth'] == a.depth)
    model = BilinearGPT(Config(**record['model_config']))
    model.load_state_dict(torch.load(a.model_dir/f'model_depth{a.depth}.pt',
                                     map_location='cpu', weights_only=True))
    model.eval()
    scores, _ = score_tables(model, a.length)
    u, s, vh = torch.linalg.svd(scores[a.head].double(), full_matrices=False)
    tokens = MultiGenreLanguage().id2token
    rows = []
    for lag in [int(x) for x in a.lags.split(',')]:
        item = dict(lag=lag, singular_value=float(s[lag, a.index]))
        for name, vec in (('query', u[lag, :, a.index]),
                          ('key', vh[lag, a.index, :])):
            top = vec.abs().topk(a.top).indices.tolist()
            item[name] = [dict(token=tokens[i], loading=float(vec[i]))
                          for i in top]
        rows.append(item)
    continuity = {}
    for name, vec in (('query', u[:, :, a.index]),
                      ('key', vh[:, a.index, :])):
        adjacent = (vec[:-1]*vec[1:]).sum(-1).abs()
        top = vec.abs().argmax(-1)
        continuity[name] = dict(median_adjacent_abs_cosine=float(adjacent.median()),
                                minimum_adjacent_abs_cosine=float(adjacent.min()),
                                adjacent_top_token_repeat_fraction=float(
                                    (top[:-1] == top[1:]).float().mean()),
                                distinct_top_tokens=int(top.unique().numel()))
    result = dict(config={**vars(a), 'model_dir': str(a.model_dir),
                          'output': str(a.output)},
                  continuity=continuity, rows=rows,
                  caveat='SVD vector signs are arbitrary; absolute loadings '
                         'and absolute adjacent cosines are sign-invariant. '
                         'Top-token lists alone do not establish semantic '
                         'meaning or causal sufficiency.')
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(continuity)


if __name__ == '__main__':
    main()

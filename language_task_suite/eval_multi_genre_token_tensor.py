"""Replay a competent multi-genre model with its weight-only layer-0 tensor.

The layer-0 input is an embedding lookup followed by per-token RMSNorm. Its
attention output is a sum of token-pair score times token-value vectors. The
pair score depends on relative lag up to the model's bfloat16 RoPE rounding.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

from eval_v6_first_layer_token_tensor import score_matrices, token_factors
from multi_genre_language import MultiGenreLanguage


@torch.no_grad()
def score_tables(model, length):
    factors, downstream = token_factors(model)
    frequency = model.blocks[0].attn.rotary.inv_freq
    tables = torch.stack([score_matrices(factors, frequency, lag, 0)[2]
                          for lag in range(length)], dim=1)
    return tables, downstream


@torch.no_grad()
def first_layer_from_table(idx, tables, downstream):
    batch, length = idx.shape
    lag = (torch.arange(length, device=idx.device)[:, None] -
           torch.arange(length, device=idx.device)[None, :]).clamp_min(0)
    scores = tables[:, lag, idx[:, :, None], idx[:, None, :]]
    scores = scores.masked_fill(~torch.ones(length, length,
        dtype=torch.bool, device=idx.device).tril()[None, None], 0.)
    values = downstream[:, idx]
    return torch.einsum('hbqk,hbkd->bqd', scores, values)


@torch.no_grad()
def forward_from_table(model, idx, tables, downstream):
    x = model.embed(idx) + first_layer_from_table(idx, tables, downstream)
    for block in model.blocks[1:]:
        x = block(x)
    if model.use_rmsnorm or model.use_final_norm:
        x = model.final_norm(x)
    return model.lm_head(x)


@torch.no_grad()
def audit(model, n, length, seed):
    assert model.config.n_layer >= 2
    assert model.blocks[0].mlp is None
    tables, downstream = score_tables(model, length)
    gen = MultiGenreLanguage(seed=seed)
    ids, _, _ = gen.sample_batch(n, length)
    idx = torch.tensor(ids, dtype=torch.long)
    native_layer = model.blocks[0].attn(model.blocks[0].norm_attn(model.embed(idx)))
    replay_layer = first_layer_from_table(idx, tables, downstream)
    native_logits = model(idx)[0]
    replay_logits = forward_from_table(model, idx, tables, downstream)
    return dict(n=n, length=length,
                max_abs_layer=float((native_layer-replay_layer).abs().max()),
                relative_l2_layer=float((native_layer-replay_layer).norm()/native_layer.norm()),
                max_abs_logit=float((native_logits-replay_logits).abs().max()),
                mean_model_to_replay_kl=float((
                    torch.softmax(native_logits.float(), -1)*(
                        torch.log_softmax(native_logits.float(), -1)-
                        torch.log_softmax(replay_logits.float(), -1))).sum(-1).mean()),
                table_shape=list(tables.shape))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dirs', nargs='+', required=True)
    p.add_argument('--depth', type=int, default=3)
    p.add_argument('--n', type=int, default=16)
    p.add_argument('--length', type=int, default=71)
    p.add_argument('--seed', type=int, default=9561)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config
    torch.set_num_threads(4)
    results = {}
    for directory in a.model_dirs:
        folder = Path(directory)
        saved = json.loads((folder/'results.json').read_text())
        record = next(row for row in saved['records'] if row['depth'] == a.depth)
        model = BilinearGPT(Config(**record['model_config']))
        model.load_state_dict(torch.load(folder/f'model_depth{a.depth}.pt',
                                         map_location='cpu', weights_only=True))
        model.eval()
        results[folder.name] = audit(model, a.n, a.length, a.seed)
        print(folder.name, results[folder.name], flush=True)
    a.output.write_text(json.dumps(dict(config={**vars(a),
                                            'output': str(a.output)},
                                        results=results,
                                        caveat='Exact token-pair algebra, with '
                                               'relative-lag approximation for '
                                               'bfloat16 RoPE. This is a layer-0 '
                                               'functional tensor, not an end-to-end '
                                               'canonical decomposition.'),
                                   indent=2)+'\n')


if __name__ == '__main__':
    main()

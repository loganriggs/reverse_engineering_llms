"""Verify an exact, weight-only token-pair tensor for v6's first attention layer.

The input to layer zero is a token embedding (plus per-token RMSNorm when
enabled), so for each relative position its whole head is a token-pair score
times a token-value vector. This is a functional object for this layer, not a
canonical decomposition of the end-to-end language model.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

from surface_mixture_v6 import SurfaceMixtureV6


def rotate(x, position, inv_freq):
    phase = position*inv_freq
    # Match Rotary.forward, which stores the trigonometric tables in bfloat16.
    cosine = phase.cos().to(torch.bfloat16).float()
    sine = phase.sin().to(torch.bfloat16).float()
    half = x.shape[-1]//2
    left, right = x[..., :half], x[..., half:]
    return torch.cat((left*cosine+right*sine,
                      -left*sine+right*cosine), dim=-1)


@torch.no_grad()
def token_factors(model):
    attention = model.blocks[0].attn
    if attention.qk_norm:
        raise ValueError('This token-table replay does not implement post-RoPE QK normalization')
    x = model.embed.weight.detach()
    if model.config.use_rmsnorm:
        x = model.blocks[0].norm_attn(x)
    h, d = attention.n_head, attention.head_dim
    project = lambda layer: layer(x).view(len(x), h, d)
    q1, k1, q2, k2, value = [project(layer) for layer in (
        attention.q1, attention.k1, attention.q2, attention.k2, attention.v)]
    # [head, key token, output coordinate]. The output matrix is applied once
    # after all heads are concatenated, but linearity permits this split.
    downstream = torch.stack([
        value[:, head] @ attention.out.weight[:, head*d:(head+1)*d].T
        for head in range(h)])
    return (q1, k1, q2, k2), downstream


@torch.no_grad()
def score_matrices(factors, inv_freq, query_pos, key_pos):
    q1, k1, q2, k2 = factors
    d = q1.shape[-1]
    a = torch.einsum('qhd,khd->hqk',
                     rotate(q1, query_pos, inv_freq),
                     rotate(k1, key_pos, inv_freq))/d
    b = torch.einsum('qhd,khd->hqk',
                     rotate(q2, query_pos, inv_freq),
                     rotate(k2, key_pos, inv_freq))/d
    return a, b, a*b


@torch.no_grad()
def replay_first_layer(model, token_ids, factors, downstream):
    attention = model.blocks[0].attn
    inv_freq = attention.rotary.inv_freq
    q1, k1, q2, k2 = factors
    n = len(token_ids)
    output = torch.zeros((n, model.config.n_embd))
    for i, query_id in enumerate(token_ids):
        for j in range(i+1):
            key_id = token_ids[j]
            s1 = (rotate(q1[query_id], i, inv_freq)*
                  rotate(k1[key_id], j, inv_freq)).sum(-1)/attention.head_dim
            s2 = (rotate(q2[query_id], i, inv_freq)*
                  rotate(k2[key_id], j, inv_freq)).sum(-1)/attention.head_dim
            output[i] += ((s1*s2)[:, None]*downstream[:, key_id]).sum(0)
    return output


@torch.no_grad()
def analyze(model, gen, n_sequences, length, lags):
    factors, downstream = token_factors(model)
    attention = model.blocks[0].attn
    ids, _, _ = gen.sample_batch(n_sequences, length)
    errors = []
    for row in ids:
        token_ids = torch.tensor(row, dtype=torch.long)
        x = model.embed(token_ids[None])
        if model.config.use_rmsnorm:
            x = model.blocks[0].norm_attn(x)
        native = attention(x)[0]
        replay = replay_first_layer(model, token_ids, factors, downstream)
        errors.append(dict(max_abs=float((native-replay).abs().max()),
                           relative_l2=float((native-replay).norm()/native.norm())))
    spectra = {}
    for lag in lags:
        first, second, product = score_matrices(
            factors, attention.rotary.inv_freq, lag, 0)
        rows = []
        for head in range(attention.n_head):
            singular = torch.linalg.svdvals(product[head].double())
            energy = singular.square()
            rows.append(dict(head=head,
                             first_rank_approx=int(torch.linalg.matrix_rank(
                                 first[head], tol=1e-5)),
                             second_rank_approx=int(torch.linalg.matrix_rank(
                                 second[head], tol=1e-5)),
                             product_energy_top={str(k): float(
                                 energy[:k].sum()/energy.sum())
                                 for k in (4, 8, 16, 32, 64)},
                             both_negative_pair_fraction=float((
                                 (first[head] < 0)&(second[head] < 0)
                                 ).float().mean()),
                             both_negative_score_energy_fraction=float((
                                 product[head][(first[head] < 0)&
                                               (second[head] < 0)].square().sum()/
                                 product[head].square().sum()))))
        spectra[str(lag)] = rows
    return dict(replay=dict(n_sequences=n_sequences, length=length,
                            max_abs=max(row['max_abs'] for row in errors),
                            mean_relative_l2=sum(row['relative_l2']
                                                 for row in errors)/len(errors)),
                relative_lag_spectra=spectra)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    parser.add_argument('--n-sequences', type=int, default=8)
    parser.add_argument('--length', type=int, default=48)
    parser.add_argument('--lags', default='0,1,4,16')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, args.source)
    from model import BilinearGPT, Config
    torch.set_num_threads(4)
    saved = json.loads((args.model_dir/'results.json').read_text())
    record = next(row for row in saved['records'] if row['depth'] == args.depth)
    model = BilinearGPT(Config(**record['model_config']))
    model.load_state_dict(torch.load(args.model_dir/f'model_depth{args.depth}.pt',
                                     map_location='cpu', weights_only=True))
    model.eval()
    gen = SurfaceMixtureV6(seed=9401, quote_rate=.30, source_rate=.30,
                           question_rate=.05, counterbalanced_queries=True)
    result = analyze(model, gen, args.n_sequences, args.length,
                     [int(x) for x in args.lags.split(',')])
    result['config'] = {**vars(args), 'model_dir': str(args.model_dir),
                        'output': str(args.output)}
    result['caveat'] = ('Exact token-pair functions of layer zero, including '
                        'RoPE and per-token RMSNorm. SVD spectra and factor '
                        'signs do not identify an end-to-end source circuit.')
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(result['replay'])

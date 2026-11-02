"""Check exact token-space QK/OV and direct-path fold of first attention block."""

import argparse
import json
import sys
from pathlib import Path

import torch

from surface_mixture_v2_varied import SurfaceMixtureV2Varied


def zero_fraction(x):
    return float((x == 0).float().mean())


def top_abs_mass(x, fraction=.01):
    values = x.detach().abs().flatten()
    count = max(1, int(fraction*len(values)))
    return float(values.topk(count).values.sum()/values.sum().clamp_min(1e-30))


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--n', type=int, default=64)
    p.add_argument('--length', type=int, default=72)
    p.add_argument('--seed', type=int, default=20261130)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config, apply_rotary_emb

    saved = json.loads((a.model_dir/'results.json').read_text())
    assert saved['config']['generator'] == 'v2_varied'
    gen = SurfaceMixtureV2Varied(seed=a.seed, **saved['generator_kwargs'])
    ids, _, _ = gen.sample_batch(a.n, a.length)
    tokens = torch.tensor(ids, dtype=torch.long)
    row = saved['records'][0]
    model = BilinearGPT(Config(**row['model_config']))
    model.load_state_dict(torch.load(a.model_dir/'model_depth2.pt',
                                     weights_only=True, map_location='cpu'))
    model.eval()
    block = model.blocks[0]
    attention = block.attn
    h, d = attention.n_head, attention.head_dim
    embeddings = model.embed.weight.detach()
    normalized = block.norm_attn(embeddings)
    factors = {name: getattr(attention, name)(normalized).reshape(-1, h, d)
               for name in ('q1', 'k1', 'q2', 'k2', 'v')}
    selected = {name: factor[tokens] for name, factor in factors.items()}
    cos, sin = attention.rotary(selected['q1'])
    scores = []
    for branch in (1, 2):
        q = apply_rotary_emb(selected[f'q{branch}'], cos, sin)
        k = apply_rotary_emb(selected[f'k{branch}'], cos, sin)
        scores.append(torch.einsum('bihd,bjhd->bhij', q, k)/d)
    mask = torch.tril(torch.ones(a.length, a.length, dtype=torch.bool))
    pattern = (scores[0]*scores[1]).masked_fill(~mask, 0)
    head_values = torch.einsum('bhij,bjhd->bihd', pattern, selected['v'])
    head_paths = torch.stack([
        head_values[:, :, head] @
        attention.out.weight[:, head*d:(head+1)*d].T
        for head in range(h)])
    folded_attention = head_paths.sum(0)
    native_embeddings = model.embed(tokens)
    native_attention = attention(block.norm_attn(native_embeddings))
    native_block = block(native_embeddings)
    errors = dict(
        attention_max_abs=float((folded_attention-native_attention).abs().max()),
        attention_relative_l2=float(torch.linalg.vector_norm(
            folded_attention-native_attention)/
            torch.linalg.vector_norm(native_attention)),
        direct_plus_heads_max_abs=float((native_embeddings+folded_attention-
                                          native_block).abs().max()),
        direct_plus_heads_relative_l2=float(torch.linalg.vector_norm(
            native_embeddings+folded_attention-native_block)/
            torch.linalg.vector_norm(native_block)))
    if errors['attention_relative_l2'] > 1e-5 or errors['direct_plus_heads_relative_l2'] > 1e-5:
        raise AssertionError(f'Token-space fold did not replay: {errors}')
    densities = dict(embedding_zero_fraction=zero_fraction(embeddings),
                     normalized_embedding_zero_fraction=zero_fraction(normalized),
                     raw_attention_weight_zero_fraction={
                         name: zero_fraction(getattr(attention, name).weight)
                         for name in ('q1', 'k1', 'q2', 'k2', 'v', 'out')},
                     token_factor_zero_fraction={
                         name: zero_fraction(factor)
                         for name, factor in factors.items()})
    # A small weight-only functional census: exact token-pair score tables at
    # lag one, and the OV token-to-residual map for each complete head.
    pair_tables = {}
    for branch in (1, 2):
        query = apply_rotary_emb(factors[f'q{branch}'][:, None],
                                 cos[:, 16:17], sin[:, 16:17])[:, 0]
        key = apply_rotary_emb(factors[f'k{branch}'][:, None],
                               cos[:, 15:16], sin[:, 15:16])[:, 0]
        pair_tables[f'qk{branch}'] = torch.einsum('vhd,uhd->hvu',
                                                 query, key)/d
    pair_tables['product'] = pair_tables['qk1']*pair_tables['qk2']
    pair_tables['ov'] = torch.stack([
        factors['v'][:, head] @
        attention.out.weight[:, head*d:(head+1)*d].T
        for head in range(h)])
    functional_tables = {name: dict(
        zero_fraction=zero_fraction(table),
        top_one_percent_abs_mass=top_abs_mass(table),
        head_zero_fraction=[zero_fraction(table[head]) for head in range(h)],
        head_top_one_percent_abs_mass=[top_abs_mass(table[head])
                                       for head in range(h)])
        for name, table in pair_tables.items()}
    observed_scores = {'qk1': scores[0], 'qk2': scores[1],
                       'product': scores[0]*scores[1]}
    observed_causal_pairs = {name: dict(
        top_one_percent_abs_mass=top_abs_mass(value[:, :, mask]),
        head_top_one_percent_abs_mass=[
            top_abs_mass(value[:, head, mask]) for head in range(h)])
        for name, value in observed_scores.items()}
    result = dict(config={**vars(a), 'model_dir': str(a.model_dir),
                          'output': str(a.output)}, errors=errors,
                  densities=densities, functional_tables=functional_tables,
                  observed_causal_pairs=observed_causal_pairs,
                  head_path_rms=[float(x.float().square().mean().sqrt())
                                 for x in head_paths],
                  caveat='This exactly folds the first layer because its '
                         'inputs are token embeddings. It keeps RoPE and RMSNorm '
                         'semantics; token-space factor density is a diagnostic, '
                         'not a human-readable rule decomposition. Later layers '
                         'receive contextual residual activations.')
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(errors, indent=2))


if __name__ == '__main__':
    main()

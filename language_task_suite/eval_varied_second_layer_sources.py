"""Decompose final-query second-layer head paths by their source positions."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from eval_surface_mixture_v2_edits import build_pairs
from surface_mixture_v2_varied import SurfaceMixtureV2Varied


@torch.no_grad()
def source_contributions(model, ids):
    from model import apply_rotary_emb

    assert model.config.n_layer == 2
    assert model.blocks[1].mlp is None
    residual = model.blocks[0](model.embed(ids))
    block = model.blocks[1]
    attn = block.attn
    x = block.norm_attn(residual)
    b, t, _ = x.shape
    h, d = attn.n_head, attn.head_dim
    project = {name: getattr(attn, name)(x).reshape(b, t, h, d)
               for name in ('q1', 'k1', 'q2', 'k2', 'v')}
    cos, sin = attn.rotary(project['q1'])
    scores = []
    for branch in (1, 2):
        q = apply_rotary_emb(project[f'q{branch}'], cos, sin)
        k = apply_rotary_emb(project[f'k{branch}'], cos, sin)
        scores.append(torch.einsum('bihd,bjhd->bhij', q, k)/d)
    pattern = scores[0][:, :, -1]*scores[1][:, :, -1]
    contributions = pattern.unsqueeze(-1)*project['v'].permute(0, 2, 1, 3)
    head_sums = contributions.sum(2).reshape(b, h*d)
    last_residual = residual[:, -1] + attn.out(head_sums)
    last_logits = model.lm_head(model.final_norm(last_residual))
    factors = dict(score1=scores[0][:, :, -1],
                   score2=scores[1][:, :, -1],
                   value=project['v'].permute(0, 2, 1, 3))
    return contributions, last_residual, last_logits, factors


def summarize(records, n_head):
    by_kind = defaultdict(list)
    for row in records:
        by_kind[row['kind']].append(row)
    result = {}
    for kind, rows in by_kind.items():
        head_rows = {}
        for head in range(n_head):
            edit = np.asarray([row['source_alignment'][head][row['edit_position']]
                               for row in rows])
            maximum = np.asarray([max(row['source_alignment'][head])
                                  for row in rows])
            top_position = np.asarray([
                int(np.argmax(row['source_alignment'][head])) for row in rows])
            edit_position = np.asarray([row['edit_position'] for row in rows])
            length = np.asarray([row['length'] for row in rows])
            head_rows[str(head)] = dict(
                mean_edit_position_alignment=float(edit.mean()),
                mean_max_source_alignment=float(maximum.mean()),
                edit_is_top_source_fraction=float(
                    (top_position == edit_position).mean()),
                median_top_source_lag=float(np.median(length-1-top_position)),
                median_edited_token_lag=float(np.median(length-1-edit_position)),
                mean_complete_head_alignment=float(np.mean([
                    row['whole_head_alignment'][head] for row in rows])))
        result[kind] = dict(
            n=len(rows),
            mean_full_effect_l1=float(np.mean([
                row['full_effect_l1'] for row in rows])),
            mean_oracle_effect_l1_error=float(np.mean([
                row['oracle_effect_l1_error'] for row in rows])),
            mean_full_oracle_effect_cosine=float(np.mean([
                row['full_oracle_effect_cosine'] for row in rows])),
            heads=head_rows)
    return result


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--n', type=int, default=64)
    p.add_argument('--length', type=int, default=72)
    p.add_argument('--seed', type=int, default=20261205)
    p.add_argument('--subject-repeat-mass', type=float)
    p.add_argument('--subject-return-mass', type=float)
    p.add_argument('--only-kind', choices=('lexical', 'induction', 'place_skip',
                                           'cue', 'nuisance_adjective',
                                           'nuisance_determiner'))
    p.add_argument('--min-source-distance', type=int)
    p.add_argument('--max-source-distance', type=int)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config

    metadata = json.loads((a.model_dir/'results.json').read_text())
    assert metadata['config']['generator'] == 'v2_varied'
    generator_kwargs = dict(metadata['generator_kwargs'])
    if a.subject_repeat_mass is not None:
        generator_kwargs['subject_repeat_mass'] = a.subject_repeat_mass
    if a.subject_return_mass is not None:
        generator_kwargs['subject_return_mass'] = a.subject_return_mass
    gen = SurfaceMixtureV2Varied(seed=a.seed, **generator_kwargs)
    pairs = build_pairs(gen, a.n, a.length)
    if a.only_kind:
        pairs = [pair for pair in pairs if pair['kind'] == a.only_kind]
    if a.min_source_distance is not None:
        pairs = [pair for pair in pairs if pair['source_distance'] is not None
                 and pair['source_distance'] >= a.min_source_distance]
    if a.max_source_distance is not None:
        pairs = [pair for pair in pairs if pair['source_distance'] is not None
                 and pair['source_distance'] <= a.max_source_distance]
    if not pairs:
        raise ValueError('No pairs meet the requested filter')
    record = next(row for row in metadata['records'] if row['depth'] == 2)
    model = BilinearGPT(Config(**record['model_config']))
    model.load_state_dict(torch.load(a.model_dir/'model_depth2.pt',
                                     weights_only=True, map_location='cpu'))
    model.eval()
    head_count = model.config.n_head
    head_width = model.config.n_embd//head_count
    out_weight = model.blocks[1].attn.out.weight.reshape(
        model.config.n_embd, head_count, head_width)
    groups = defaultdict(list)
    for index, pair in enumerate(pairs):
        groups[len(pair['original'])].append(index)
    records = [None]*len(pairs)
    max_native_logit_error = 0.
    max_attention_replay_error = 0.
    for indices in groups.values():
        for start in range(0, len(indices), 32):
            batch_indices = indices[start:start+32]
            x0 = torch.tensor([pairs[i]['original'] for i in batch_indices])
            x1 = torch.tensor([pairs[i]['edited'] for i in batch_indices])
            differences = x0 != x1
            if not bool((differences.sum(-1) == 1).all()):
                raise AssertionError('Expected exactly one edited token')
            edit_positions = differences.int().argmax(-1).numpy()
            c0, r0, logit0, _ = source_contributions(model, x0)
            c1, _, logit1, _ = source_contributions(model, x1)
            native0 = model(x0)[0][:, -1]
            native1 = model(x1)[0][:, -1]
            max_native_logit_error = max(max_native_logit_error,
                float((logit0-native0).abs().max()),
                float((logit1-native1).abs().max()))
            # The source-path sum must reproduce the complete attention output.
            attn = model.blocks[1].attn
            residual = model.blocks[0](model.embed(x0))
            native_attention = attn(model.blocks[1].norm_attn(residual))[:, -1]
            replay_attention = attn.out(c0.sum(2).reshape(len(x0), -1))
            max_attention_replay_error = max(max_attention_replay_error,
                float((replay_attention-native_attention).abs().max()))
            full = F.softmax(logit1, -1)-F.softmax(logit0, -1)
            oracle = np.zeros((len(batch_indices), len(gen.vocab)), np.float32)
            for j, index in enumerate(batch_indices):
                for token, mass in pairs[index]['q_after'].items():
                    oracle[j, gen.token2id[token]] += mass
                for token, mass in pairs[index]['q_before'].items():
                    oracle[j, gen.token2id[token]] -= mass
            oracle = torch.from_numpy(oracle)
            full_l1 = full.abs().sum(-1)
            oracle_error = (full-oracle).abs().sum(-1)
            oracle_cosine = ((full*oracle).sum(-1)/(
                full.square().sum(-1).sqrt()*oracle.square().sum(-1).sqrt()
                ).clamp_min(1e-20))
            denominator = full.square().sum(-1).clamp_min(1e-20)
            delta = c1-c0
            delta_residual = torch.einsum('bhtd,chd->bhtc', delta, out_weight)
            candidates = model.lm_head(model.final_norm(
                r0[:, None, None, :]+delta_residual))
            effects = F.softmax(candidates, -1)-F.softmax(
                logit0[:, None, None, :], -1)
            source_alignment = ((effects*full[:, None, None, :]).sum(-1)/
                                denominator[:, None, None]).numpy()
            whole_delta = delta.sum(2)
            whole_residual = torch.einsum('bhd,chd->bhc',
                                          whole_delta, out_weight)
            whole_logits = model.lm_head(model.final_norm(
                r0[:, None, :]+whole_residual))
            whole_effect = F.softmax(whole_logits, -1)-F.softmax(
                logit0[:, None, :], -1)
            whole_alignment = ((whole_effect*full[:, None, :]).sum(-1)/
                               denominator[:, None]).numpy()
            for j, index in enumerate(batch_indices):
                records[index] = dict(kind=pairs[index]['kind'],
                    length=len(pairs[index]['original']),
                    edit_position=int(edit_positions[j]),
                    full_effect_l1=float(full_l1[j]),
                    oracle_effect_l1_error=float(oracle_error[j]),
                    full_oracle_effect_cosine=float(oracle_cosine[j]),
                    source_alignment=source_alignment[j].tolist(),
                    whole_head_alignment=whole_alignment[j].tolist())
    if max_native_logit_error > 2e-4 or max_attention_replay_error > 2e-4:
        raise AssertionError((max_native_logit_error,
                              max_attention_replay_error))
    result = dict(config={**vars(a), 'model_dir': str(a.model_dir),
                          'output': str(a.output)}, n_pairs=len(pairs),
                  generator_kwargs=generator_kwargs,
                  max_native_logit_error=max_native_logit_error,
                  max_attention_replay_error=max_attention_replay_error,
                  summary=summarize(records, head_count),
                  caveat='Source paths are exact at the final query of the '
                         'second attention block. Each edited source '
                         'contribution is patched individually into the '
                         'original final residual; nonlinear probabilities '
                         'mean effects do not sum. The edited computation '
                         'also changes query/key context at other sources.')
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print('pairs', len(pairs), 'logit replay', max_native_logit_error,
          'attention replay', max_attention_replay_error, flush=True)
    for kind in ('lexical', 'induction', 'place_skip', 'cue'):
        if kind not in result['summary']:
            continue
        print(kind, {h: round(row['mean_edit_position_alignment'], 3)
                     for h, row in result['summary'][kind]['heads'].items()},
              flush=True)


if __name__ == '__main__':
    main()

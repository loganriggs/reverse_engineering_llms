"""Fit compact final-readout bases across residual and attention paths.

Whole verb/object induction edit families are withheld from basis training and
development. Compare a rank-16 function-collapsed residual basis with a
rank-8+8 path-separated basis at the same total latent-coordinate budget.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from eval_v3_rule_holdout_bond import (HELD_OUT, capture, orthogonal_columns,
                                       panel, remove_duplicates)
from surface_mixture_v3 import SurfaceMixtureV3


def readout(model, residual):
    if model.use_rmsnorm or model.use_final_norm:
        residual = model.final_norm(residual)
    return model.lm_head(residual).softmax(-1)


@torch.no_grad()
def prepare(model, pairs, vocab):
    h0, z0, p0 = capture(model, [pair['original'] for pair in pairs])
    h1, z1, p1 = capture(model, [pair['edited'] for pair in pairs])
    w = model.blocks[-1].attn.out.weight
    r0 = h0 + F.linear(z0, w)
    dr_h = h1-h0
    dr_z = z1-z0
    dr = dr_h + F.linear(dr_z, w)
    replay_error = max((readout(model, r0)-p0).abs().max().item(),
                       (readout(model, r0+dr)-p1).abs().max().item())
    if replay_error > 1e-5:
        raise AssertionError(f'Endpoint replay failed: {replay_error}')
    oracle = torch.zeros_like(p0)
    for i, pair in enumerate(pairs):
        for token, mass in pair['q_after'].items():
            oracle[i, vocab[token]] += mass
        for token, mass in pair['q_before'].items():
            oracle[i, vocab[token]] -= mass
    return dict(r0=r0, dh=dr_h, dz=dr_z, dr=dr, p0=p0,
                native=p1-p0, oracle=oracle,
                kinds=[pair['kind'] for pair in pairs],
                replay_error=replay_error)


def pca(x, rank):
    _, _, vh = torch.linalg.svd(x.double(), full_matrices=False)
    return vh[:rank].T.float().contiguous()


@torch.no_grad()
def task_blind_residuals(model, seed, streams, length, positions):
    """Sample final residual states from ordinary streams, without site labels."""
    gen = SurfaceMixtureV3(seed=seed, object_vocab_size=8)
    ids, _, _ = gen.sample_batch(streams, length)
    states = []
    for start in range(0, streams, 32):
        captured = []
        hook = model.final_norm.register_forward_pre_hook(
            lambda _module, inputs: captured.append(inputs[0].detach()))
        model(torch.tensor(ids[start:start+32], dtype=torch.long))
        hook.remove()
        states.append(captured[0].reshape(-1, model.config.n_embd))
    all_states = torch.cat(states)
    rng = np.random.default_rng(seed+17)
    chosen = rng.choice(len(all_states), size=min(positions,len(all_states)),
                        replace=False)
    return all_states[chosen]


def project(x, basis):
    return (x @ basis) @ basis.T


def projected_delta(model, data, bases):
    kind, basis = bases
    if kind == 'combined':
        return project(data['dr'], basis)
    if kind == 'separate':
        h_basis, z_basis = basis
        return project(data['dh'], h_basis) + F.linear(
            project(data['dz'], z_basis), model.blocks[-1].attn.out.weight)
    raise ValueError(kind)


def effects(model, data, bases):
    delta = projected_delta(model, data, bases)
    sufficient = readout(model, data['r0']+delta)-data['p0']
    remaining = readout(model, data['r0']+data['dr']-delta)-data['p0']
    return sufficient, remaining, delta


def family_loss(model, data, bases, target):
    effect = effects(model, data, bases)[0]
    labels = sorted(set(data['kinds']))
    losses = []
    for label in labels:
        mask = torch.tensor([kind == label for kind in data['kinds']])
        losses.append((effect[mask]-data[target][mask]).square().sum(-1).mean())
    return torch.stack(losses).mean()


def optimize(model, fit, dev, initialization, target, steps, lr):
    kind, basis = initialization
    raw = ([torch.nn.Parameter(b.clone()) for b in basis] if kind == 'separate'
           else [torch.nn.Parameter(basis.clone())])
    optimizer = torch.optim.Adam(raw, lr=lr)
    best = (float('inf'), None, -1)
    for step in range(steps+1):
        current = (kind, tuple(orthogonal_columns(b) for b in raw)
                   if kind == 'separate' else orthogonal_columns(raw[0]))
        with torch.no_grad():
            dev_loss = family_loss(model, dev, current, target).item()
            if dev_loss < best[0]:
                snapshot = (kind, tuple(b.detach().clone() for b in current[1])
                            if kind == 'separate' else current[1].detach().clone())
                best = (dev_loss, snapshot, step)
        if step == steps:
            break
        optimizer.zero_grad()
        family_loss(model, fit, current, target).backward()
        optimizer.step()
    return best[1], dict(best_dev_squared_effect=best[0], best_step=best[2],
                         objective=target)


@torch.no_grad()
def score(model, data, bases):
    sufficient, remaining, delta = effects(model, data, bases)
    result = {}
    for label in sorted(set(data['kinds'])):
        mask = torch.tensor([kind == label for kind in data['kinds']])
        l1 = lambda x: x[mask].abs().sum(-1).mean().item()
        result[label] = dict(
            n=int(mask.sum()), native_effect_l1=l1(data['native']),
            oracle_effect_l1=l1(data['oracle']),
            sufficient_effect_l1=l1(sufficient),
            sufficient_to_native_l1=l1(sufficient-data['native']),
            sufficient_to_oracle_l1=l1(sufficient-data['oracle']),
            native_to_oracle_l1=l1(data['native']-data['oracle']),
            remaining_effect_l1=l1(remaining),
            remaining_to_zero_l1=l1(remaining),
            residual_delta_mse=(data['dr'][mask]-delta[mask]).square().sum(-1).mean().item(),
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    parser.add_argument('--model-dir', default='language_task_suite/surface_mixture_v3_object8')
    parser.add_argument('--seed', type=int, default=30101)
    parser.add_argument('--test-seed', type=int, default=31001)
    parser.add_argument('--streams', type=int, default=160)
    parser.add_argument('--fit-cap', type=int, default=120)
    parser.add_argument('--dev-cap', type=int, default=60)
    parser.add_argument('--test-cap', type=int, default=160)
    parser.add_argument('--length', type=int, default=72)
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--lr', type=float, default=.03)
    parser.add_argument('--blind-streams', type=int, default=128)
    parser.add_argument('--blind-positions', type=int, default=4096)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.rank % 2:
        parser.error('--rank must be even for equal path allocation')
    torch.set_num_threads(4)
    sys.path.insert(0, args.source)
    from model import BilinearGPT, Config
    folder = Path(args.model_dir)
    record = json.loads((folder/'results.json').read_text())['records'][0]
    model = BilinearGPT(Config(**record['model_config']))
    model.load_state_dict(torch.load(folder/f"model_depth{record['depth']}.pt",
                                     weights_only=True, map_location='cpu'))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    vocab = SurfaceMixtureV3(object_vocab_size=8).token2id
    raw_fit = [pair for pair in panel(args.seed,args.streams,args.fit_cap,args.length)
               if pair['kind'] not in HELD_OUT]
    raw_dev = [pair for pair in panel(args.seed+1,args.streams,args.dev_cap,args.length)
               if pair['kind'] not in HELD_OUT]
    raw_test = panel(args.test_seed,args.streams*2,args.test_cap,args.length)
    seen = set()
    fit_pairs, removed_fit = remove_duplicates(raw_fit, seen)
    dev_pairs, removed_dev = remove_duplicates(raw_dev, seen)
    test_pairs, removed_test = remove_duplicates(raw_test, seen)
    assert not any(pair['kind'] in HELD_OUT for pair in fit_pairs+dev_pairs)
    assert all(any(pair['kind'] == kind for pair in test_pairs) for kind in HELD_OUT)
    fit = prepare(model, fit_pairs, vocab)
    dev = prepare(model, dev_pairs, vocab)
    test = prepare(model, test_pairs, vocab)
    rank = args.rank
    half = rank//2
    d = model.config.n_embd
    combined_pca = ('combined', pca(fit['dr'], rank))
    blind_states = task_blind_residuals(model, args.seed+500,
                                         args.blind_streams,args.length,
                                         args.blind_positions)
    _, _, vh = torch.linalg.svd(model.lm_head.weight.double(), full_matrices=False)
    bases = dict(
        zero=('combined', torch.empty(d,0)),
        full=('combined',torch.eye(d)),
        combined_delta_pca=combined_pca,
        task_blind_stream_pca=('combined',pca(blind_states,rank)),
        combined_context_pca=('combined',pca(torch.cat((fit['r0'],fit['r0']+fit['dr'])),rank)),
        combined_lm_head_svd=('combined',vh[:rank].T.float().contiguous()),
        separated_delta_pca=('separate',(pca(fit['dh'],half),pca(fit['dz'],half))),
    )
    metadata = {}
    for name, initial, target in (
        ('combined_teacher_optimized',combined_pca,'native'),
        ('combined_oracle_optimized',combined_pca,'oracle'),
        ('separated_teacher_optimized',bases['separated_delta_pca'],'native'),
        ('separated_oracle_optimized',bases['separated_delta_pca'],'oracle'),
    ):
        bases[name], metadata[name] = optimize(model, fit, dev, initial,
                                                target, args.steps, args.lr)
        print(name, metadata[name], flush=True)
    rows = {name: score(model, test, basis) for name, basis in bases.items()}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    basis_path = path.with_suffix('.pt')
    torch.save({name: basis for name, basis in bases.items()}, basis_path)
    output = dict(config=vars(args), heldout_fit_families=HELD_OUT,
                  basis_checkpoint=str(basis_path),
                  fit_counts=dict(Counter(pair['kind'] for pair in fit_pairs)),
                  dev_counts=dict(Counter(pair['kind'] for pair in dev_pairs)),
                  test_counts=dict(Counter(pair['kind'] for pair in test_pairs)),
                  duplicates_removed=dict(fit=removed_fit,dev=removed_dev,test=removed_test),
                  max_endpoint_replay_error=max(fit['replay_error'],dev['replay_error'],
                                                test['replay_error']),
                  blind_positions_used=len(blind_states),
                  optimization=metadata, rows=rows,
                  caveat='The frozen teacher saw all rule families. Fit/dev omit '
                         'induction edits only for featurizer learning. Combined '
                         'bases collapse paths in output residual space; separate '
                         'bases retain h and z coordinates. Sufficient and '
                         'remaining effects are cross-world final-readout '
                         'interventions, not unique circuit attributions.')
    path.write_text(json.dumps(output, indent=2)+'\n')
    for name, groups in rows.items():
        print(name, {kind: round(groups[kind]['sufficient_to_native_l1'],4)
                     for kind in HELD_OUT}, flush=True)


if __name__ == '__main__':
    main()

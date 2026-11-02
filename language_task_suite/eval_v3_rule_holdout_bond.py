"""Compare last-attention-bond bases with whole edit-rule-family fit holdouts.

The model is frozen. Only a rank-k orthogonal bond basis is learned. The
teacher-local objective matches the full-width bond-patch probability effect,
not the donor's total effect (which can use other paths). Fit and development
omit both induction edit families; test contains fresh instances of all rules.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from eval_surface_mixture_v2_edits import build_pairs as verb_pairs
from eval_surface_mixture_v3_edits import build_object_pairs
from surface_mixture_v3 import SurfaceMixtureV3


HELD_OUT = ('induction', 'object_induction')


def panel(seed, streams, cap, length):
    pairs = (verb_pairs(SurfaceMixtureV3(seed=seed, object_vocab_size=8), streams, length)
             + build_object_pairs(SurfaceMixtureV3(seed=seed, object_vocab_size=8), streams, length))
    rng = np.random.default_rng(seed + 7721)
    groups = defaultdict(list)
    for pair in pairs:
        groups[pair['kind']].append(pair)
    return [pair for kind in sorted(groups) for pair in
            (rng.choice(groups[kind], size=min(cap, len(groups[kind])), replace=False)
             if len(groups[kind]) > cap else groups[kind])]


def remove_duplicates(pairs, used):
    kept = []
    n_removed = 0
    for pair in pairs:
        key = (pair['kind'], tuple(pair['original']), tuple(pair['edited']))
        if key in used:
            n_removed += 1
        else:
            used.add(key)
            kept.append(pair)
    return kept, n_removed


@torch.no_grad()
def capture(model, prefixes, batch_size=32):
    """Capture last-position input and pre-output bond of the final block."""
    bond = model.blocks[-1].attn.out
    block = model.blocks[-1]
    d = model.config.n_embd
    residual = torch.empty((len(prefixes), d))
    activations = torch.empty_like(residual)
    probabilities = torch.empty((len(prefixes), model.config.vocab_size))
    groups = defaultdict(list)
    for i, prefix in enumerate(prefixes):
        groups[len(prefix)].append(i)
    for indices in groups.values():
        for start in range(0, len(indices), batch_size):
            selected = indices[start:start + batch_size]
            values = {}
            hook_h = block.register_forward_pre_hook(
                lambda _module, inputs: values.__setitem__('h', inputs[0][:, -1].detach()))
            hook_z = bond.register_forward_pre_hook(
                lambda _module, inputs: values.__setitem__('z', inputs[0][:, -1].detach()))
            x = torch.tensor([prefixes[i] for i in selected], dtype=torch.long)
            logits = model(x)[0][:, -1]
            hook_h.remove()
            hook_z.remove()
            residual[selected] = values['h']
            activations[selected] = values['z']
            probabilities[selected] = logits.softmax(-1)
    return residual, activations, probabilities


def readout(model, h, z):
    x = h + F.linear(z, model.blocks[-1].attn.out.weight)
    if model.use_rmsnorm or model.use_final_norm:
        x = model.final_norm(x)
    return model.lm_head(x).softmax(-1)


def prepare(model, pairs, vocab):
    h0, z0, p0 = capture(model, [p['original'] for p in pairs])
    _, z1, p1 = capture(model, [p['edited'] for p in pairs])
    with torch.no_grad():
        replay_error = (readout(model, h0, z0) - p0).abs().max().item()
        if replay_error > 1e-5:
            raise AssertionError(f'Local readout does not replay teacher: {replay_error}')
        full = readout(model, h0, z1) - p0
    oracle = torch.zeros_like(full)
    for i, pair in enumerate(pairs):
        for token, value in pair['q_after'].items():
            oracle[i, vocab[token]] += value
        for token, value in pair['q_before'].items():
            oracle[i, vocab[token]] -= value
    return dict(h=h0, z=z0, dz=z1-z0, p0=p0, donor_effect=p1-p0,
                full_effect=full, oracle_effect=oracle,
                kinds=[p['kind'] for p in pairs], replay_error=replay_error)


def orthogonal_columns(a):
    return torch.linalg.qr(a).Q


def baselines(fit, weight, rank, seed):
    d = fit['z'].shape[-1]
    def pca(x):
        _, _, vh = torch.linalg.svd(x.double(), full_matrices=False)
        return vh[:rank].T.float().contiguous()
    _, _, vh = torch.linalg.svd(weight.double(), full_matrices=False)
    generator = torch.Generator().manual_seed(seed)
    return dict(
        zero=torch.empty(d, 0),
        contrast_pca=pca(fit['dz']),
        context_pca=pca(torch.cat((fit['z'], fit['z'] + fit['dz']))),
        weight_svd=vh[:rank].T.float().contiguous(),
        random=orthogonal_columns(torch.randn(d, rank, generator=generator)),
        full=torch.eye(d),
    )


def projected_effect(model, data, basis):
    dz = (data['dz'] @ basis) @ basis.T
    return readout(model, data['h'], data['z'] + dz) - data['p0']


def family_mean_loss(model, data, basis, kinds):
    effect = projected_effect(model, data, basis)
    losses = []
    for kind in kinds:
        mask = torch.tensor([label == kind for label in data['kinds']])
        losses.append((effect[mask] - data['full_effect'][mask]).square().sum(-1).mean())
    return torch.stack(losses).mean()


def optimize_basis(model, fit, dev, initial, steps, lr):
    kinds = sorted(set(fit['kinds']))
    assert not set(HELD_OUT).intersection(kinds)
    raw = torch.nn.Parameter(initial.clone())
    opt = torch.optim.Adam([raw], lr=lr)
    best = (float('inf'), initial.clone(), -1)
    for step in range(steps + 1):
        with torch.no_grad():
            basis = orthogonal_columns(raw)
            loss = family_mean_loss(model, dev, basis, kinds).item()
            if loss < best[0]:
                best = (loss, basis.detach().clone(), step)
        if step == steps:
            break
        opt.zero_grad()
        loss = family_mean_loss(model, fit, orthogonal_columns(raw), kinds)
        loss.backward()
        opt.step()
    return best[1], dict(best_dev_squared_effect=best[0], best_step=best[2])


@torch.no_grad()
def scores(model, data, basis):
    effect = projected_effect(model, data, basis)
    result = {}
    for kind in sorted(set(data['kinds'])):
        mask = torch.tensor([label == kind for label in data['kinds']])
        def l1(x):
            return x[mask].abs().sum(-1).mean().item()
        result[kind] = dict(n=int(mask.sum()),
                            oracle_effect_l1=l1(data['oracle_effect']),
                            donor_effect_l1=l1(data['donor_effect']),
                            full_effect_l1=l1(data['full_effect']),
                            projected_effect_l1=l1(effect),
                            projected_to_full_l1=l1(effect-data['full_effect']),
                            full_to_donor_l1=l1(data['full_effect']-data['donor_effect']),
                            projected_to_donor_l1=l1(effect-data['donor_effect']),
                            projected_to_oracle_l1=l1(effect-data['oracle_effect']),
                            donor_to_oracle_l1=l1(data['donor_effect']-data['oracle_effect']),
                            projected_activation_mse=(
                                (data['dz'][mask] -
                                 (data['dz'][mask] @ basis) @ basis.T).square().sum(-1).mean().item()))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dir', default='language_task_suite/surface_mixture_v3_object8')
    p.add_argument('--streams', type=int, default=96)
    p.add_argument('--length', type=int, default=72)
    p.add_argument('--fit-cap', type=int, default=80)
    p.add_argument('--dev-cap', type=int, default=40)
    p.add_argument('--test-cap', type=int, default=100)
    p.add_argument('--rank', type=int, default=8)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--lr', type=float, default=.03)
    p.add_argument('--seed', type=int, default=27001)
    p.add_argument('--test-seed', type=int, default=None,
                   help='Independent test generator seed; default is seed+2')
    p.add_argument('--output', required=True)
    args = p.parse_args()
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
    gen = SurfaceMixtureV3(object_vocab_size=8)

    raw_fit = [x for x in panel(args.seed, args.streams, args.fit_cap, args.length)
               if x['kind'] not in HELD_OUT]
    raw_dev = [x for x in panel(args.seed+1, args.streams, args.dev_cap, args.length)
               if x['kind'] not in HELD_OUT]
    test_seed = args.seed+2 if args.test_seed is None else args.test_seed
    raw_test = panel(test_seed, args.streams*2, args.test_cap, args.length)
    seen = set()
    fit, fit_removed = remove_duplicates(raw_fit, seen)
    dev, dev_removed = remove_duplicates(raw_dev, seen)
    test, test_removed = remove_duplicates(raw_test, seen)
    assert not any(pair['kind'] in HELD_OUT for pair in fit + dev)
    assert all(any(pair['kind'] == kind for pair in test) for kind in HELD_OUT)
    print('Pairs', {name: dict(Counter(p['kind'] for p in pairs))
                    for name, pairs in (('fit',fit), ('dev',dev), ('test',test))}, flush=True)
    fit_data = prepare(model, fit, gen.token2id)
    dev_data = prepare(model, dev, gen.token2id)
    test_data = prepare(model, test, gen.token2id)
    basis_map = baselines(fit_data, model.blocks[-1].attn.out.weight,
                          args.rank, args.seed)
    optimized, meta = optimize_basis(model, fit_data, dev_data,
                                     basis_map['contrast_pca'], args.steps, args.lr)
    basis_map['causal_effect_optimized'] = optimized
    rows = {name: scores(model, test_data, basis) for name, basis in basis_map.items()}
    result = dict(config=vars(args), heldout_fit_families=HELD_OUT,
                  fit_counts=dict(Counter(p['kind'] for p in fit)),
                  dev_counts=dict(Counter(p['kind'] for p in dev)),
                  test_counts=dict(Counter(p['kind'] for p in test)),
                  duplicates_removed=dict(fit=fit_removed,dev=dev_removed,test=test_removed),
                  max_replay_error=max(fit_data['replay_error'],dev_data['replay_error'],
                                       test_data['replay_error']),
                  optimization=meta, rows=rows,
                  caveat='This is a local final-bond mediation experiment. The frozen '
                         'teacher saw all rule families in language training. Held-out '
                         'families were excluded only from basis fit/dev. A successful '
                         'basis does not establish a unique or human-readable circuit.')
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2)+'\n')
    for name, groups in rows.items():
        print(name, {k: round(groups[k]['projected_to_full_l1'], 4)
                     for k in HELD_OUT}, flush=True)


if __name__ == '__main__':
    main()

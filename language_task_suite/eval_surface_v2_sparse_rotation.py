"""Unlabeled sparse rotations of a weight-derived attention bond subspace.

Quartimax fits an orthogonal rotation that concentrates each observation in
few coordinates. We fit it either on unlabeled causal response differences or
on ordinary activations. Rule labels are used afterward only to name axes and
evaluate transfer to disjoint sentence frames.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from eval_surface_mixture_v2_edits import build_pairs
from eval_surface_v2_rule_subspaces import (FIT_FRAMES, KINDS, TEST_FRAMES,
                                            response_deltas)
from surface_mixture_v2 import SurfaceMixtureV2
from train_surface_mixture import make_batch


def unit_rows(x):
    x = x.double()
    return x/x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def quartimax(x, starts=4, iterations=100, seed=4101):
    """Orthogonal row-sparsity rotation via repeated polar updates."""
    x = unit_rows(x)
    d = x.shape[1]
    gen = torch.Generator().manual_seed(seed)
    best = torch.eye(d, dtype=torch.float64)
    best_objective = float(x.pow(4).sum(-1).mean())
    for start in range(starts):
        q = (torch.eye(d, dtype=torch.float64) if start == 0 else
             torch.linalg.qr(torch.randn(d, d, generator=gen,
                                         dtype=torch.float64))[0])
        for _ in range(iterations):
            y = x@q
            u, _, vh = torch.linalg.svd(x.T@(y**3), full_matrices=False)
            new_q = u@vh
            if torch.linalg.matrix_norm(new_q-q) < 1e-8:
                q = new_q
                break
            q = new_q
        objective = float((x@q).pow(4).sum(-1).mean())
        if objective > best_objective:
            best, best_objective = q, objective
    assert torch.linalg.matrix_norm(best.T@best-torch.eye(d)) < 1e-7
    return best.float(), best_objective


def classification(fit_delta, fit_kinds, test_delta, test_kinds, basis):
    fit_z = unit_rows(fit_delta@basis)
    test_z = unit_rows(test_delta@basis)
    fit_energy = torch.stack([fit_z[[k == kind for k in fit_kinds]].square().mean(0)
                              for kind in KINDS])
    axis_kind = fit_energy.argmax(0)
    predicted = axis_kind[test_z.abs().argmax(-1)]
    actual = torch.tensor([KINDS.index(kind) for kind in test_kinds])
    counts = {}
    for i, kind in enumerate(KINDS):
        mask = actual == i
        counts[kind] = dict(n=int(mask.sum()), accuracy=float((predicted[mask] == i).float().mean()))
    top8 = {}
    for i, kind in enumerate(KINDS):
        axes = fit_energy[i].topk(8).indices
        mask = actual == i
        top8[kind] = float(test_z[mask][:, axes].square().sum(-1).mean())
    return dict(macro_accuracy=float(np.mean([v['accuracy'] for v in counts.values()])),
                per_kind=counts, top8_energy=top8,
                axis_kind_counts={kind: int((axis_kind == i).sum())
                                  for i, kind in enumerate(KINDS)},
                test_quartic_sparsity=float(test_z.pow(4).sum(-1).mean()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dir', default='language_task_suite/surface_mixture_v2_soft_models')
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--layer', type=int, default=1)
    p.add_argument('--prior-width', type=int, default=32)
    p.add_argument('--nfit', type=int, default=256)
    p.add_argument('--ntest', type=int, default=256)
    p.add_argument('--length', type=int, default=72)
    p.add_argument('--reverse-frames', action='store_true')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config

    saved = json.loads((Path(a.model_dir)/'results.json').read_text())
    record = next(row for row in saved['records'] if row['depth'] == a.depth)
    model = BilinearGPT(Config(**record['model_config']))
    model.load_state_dict(torch.load(Path(a.model_dir)/f'model_depth{a.depth}.pt',
                                     map_location='cpu', weights_only=True))
    model.eval()
    bond = model.blocks[a.layer].attn.out
    _, _, vh = torch.linalg.svd(bond.weight.detach().double(), full_matrices=False)
    prior = vh[:a.prior_width].T.float()
    fit_frames = TEST_FRAMES if a.reverse_frames else FIT_FRAMES
    test_frames = FIT_FRAMES if a.reverse_frames else TEST_FRAMES
    fit = [pair for pair in build_pairs(SurfaceMixtureV2(seed=9901), a.nfit,
                                       a.length) if pair['frame'] in fit_frames
           and pair['kind'] in KINDS]
    test = [pair for pair in build_pairs(SurfaceMixtureV2(seed=8701), a.ntest,
                                        a.length) if pair['frame'] in test_frames
            and pair['kind'] in KINDS]
    fit_delta = response_deltas(model, bond, fit)@prior
    test_delta = response_deltas(model, bond, test)@prior
    fit_kinds = [pair['kind'] for pair in fit]
    test_kinds = [pair['kind'] for pair in test]
    xfit, _ = make_batch(SurfaceMixtureV2(seed=8501), 256, a.length, 'cpu')
    captured = []
    def collect(_, inputs):
        captured.append(inputs[0].detach())
    handle = bond.register_forward_pre_hook(collect)
    with torch.no_grad():
        for x in xfit.split(32):
            model(x)
    handle.remove()
    activations = torch.cat(captured).reshape(-1, model.config.n_embd)@prior
    activation_rows = activations.double()-activations.double().mean(0)
    activation_rows = activation_rows[:4096]
    response_rotation, response_objective = quartimax(fit_delta)
    usage_rotation, usage_objective = quartimax(activation_rows)
    gen = torch.Generator().manual_seed(7141)
    random_rotation = torch.linalg.qr(torch.randn(a.prior_width, a.prior_width,
                                                  generator=gen))[0]
    rotations = dict(raw_weight=torch.eye(a.prior_width),
                     random=random_rotation,
                     usage_quartimax=usage_rotation,
                     response_quartimax=response_rotation)
    results = {name: classification(fit_delta, fit_kinds, test_delta,
                                    test_kinds, rotation)
               for name, rotation in rotations.items()}
    output = dict(config=vars(a), fit_frames=fit_frames,
                  test_frames=test_frames,
                  fit_counts={kind: fit_kinds.count(kind) for kind in KINDS},
                  test_counts={kind: test_kinds.count(kind) for kind in KINDS},
                  fit_objectives=dict(response_quartimax=response_objective,
                                      usage_quartimax=usage_objective),
                  results=results,
                  caveat='The rotations use no edit-kind labels, but naming axes '
                         'by training-label response energy is supervised. A '
                         'largest-coordinate classifier may exploit token identity '
                         'instead of computation. This tests axis organization '
                         'inside a fixed weight-only subspace, not causal fidelity '
                         'or discovery of a genuinely new task.')
    Path(a.output).write_text(json.dumps(output, indent=2)+'\n')
    for name, row in results.items():
        print(name, 'macro', round(row['macro_accuracy'], 3),
              'sparsity', round(row['test_quartic_sparsity'], 3),
              'top8', {k: round(v, 3) for k, v in row['top8_energy'].items()})


if __name__ == '__main__':
    main()

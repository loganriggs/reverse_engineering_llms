"""Test whether causal-edit response subspaces transfer across sentence frames.

Fit only on lead/subject/at frames from one seed. Evaluate on disjoint
when/in-place/during frames from another seed. The measured vector is the
change at the final prefix position immediately before an attention output
projection; this is local response geometry, not a recovered circuit.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from eval_surface_mixture_v2_edits import build_pairs
from surface_mixture_v2 import SurfaceMixtureV2
from train_surface_mixture import make_batch


FIT_FRAMES = ('lead', 'subject', 'at')
TEST_FRAMES = ('when', 'in_place', 'during')
KINDS = ('lexical', 'induction', 'place_skip', 'nuisance_adjective')


@torch.no_grad()
def capture(model, bond, prefixes):
    out = torch.empty((len(prefixes), model.config.n_embd), dtype=torch.float32)
    groups = defaultdict(list)
    for i, prefix in enumerate(prefixes):
        groups[len(prefix)].append(i)
    collected = []

    def hook(_, inputs):
        collected.append(inputs[0][:, -1].detach().clone())

    handle = bond.register_forward_pre_hook(hook)
    try:
        for indices in groups.values():
            for start in range(0, len(indices), 64):
                sub = indices[start:start+64]
                x = torch.tensor([prefixes[i] for i in sub], dtype=torch.long)
                model(x)
                values = collected.pop()
                out[sub] = values
    finally:
        handle.remove()
    return out


def response_deltas(model, bond, pairs):
    before = capture(model, bond, [pair['original'] for pair in pairs])
    after = capture(model, bond, [pair['edited'] for pair in pairs])
    return after-before


def normalized_svd_basis(rows, width):
    norms = torch.linalg.vector_norm(rows, dim=-1)
    kept = rows[norms > 1e-8]/norms[norms > 1e-8, None]
    if len(kept) < width:
        raise ValueError(f'Only {len(kept)} nonzero rows for width {width}')
    _, _, vh = torch.linalg.svd(kept.double(), full_matrices=False)
    return vh[:width].T.float()


def energy(rows, basis):
    norm2 = rows.square().sum(-1)
    nonzero = norm2 > 1e-16
    ratio = ((rows[nonzero]@basis).square().sum(-1)/norm2[nonzero])
    return dict(n=int(nonzero.sum()), mean=float(ratio.mean()),
                median=float(ratio.median()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dir', default='language_task_suite/surface_mixture_v2_soft_models')
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--layer', type=int, default=0)
    p.add_argument('--nfit', type=int, default=256)
    p.add_argument('--ntest', type=int, default=256)
    p.add_argument('--length', type=int, default=72)
    p.add_argument('--widths', default='4,8,16')
    p.add_argument('--reverse-frames', action='store_true')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config

    saved = json.loads((Path(a.model_dir)/'results.json').read_text())
    record = next(r for r in saved['records'] if r['depth'] == a.depth)
    model = BilinearGPT(Config(**record['model_config']))
    model.load_state_dict(torch.load(Path(a.model_dir)/f'model_depth{a.depth}.pt',
                                     map_location='cpu', weights_only=True))
    model.eval()
    bond = model.blocks[a.layer].attn.out
    fit_frames = TEST_FRAMES if a.reverse_frames else FIT_FRAMES
    test_frames = FIT_FRAMES if a.reverse_frames else TEST_FRAMES
    fit = [pair for pair in build_pairs(SurfaceMixtureV2(seed=9901), a.nfit,
                                       a.length) if pair['frame'] in fit_frames]
    test = [pair for pair in build_pairs(SurfaceMixtureV2(seed=8701), a.ntest,
                                        a.length) if pair['frame'] in test_frames]
    fit_delta = response_deltas(model, bond, fit)
    test_delta = response_deltas(model, bond, test)
    fit_by_kind = {kind: fit_delta[[i for i, pair in enumerate(fit)
                                   if pair['kind'] == kind]] for kind in KINDS}
    test_by_kind = {kind: test_delta[[i for i, pair in enumerate(test)
                                     if pair['kind'] == kind]] for kind in KINDS}
    weight_basis = torch.linalg.svd(bond.weight.detach().double(),
                                   full_matrices=False).Vh.T.float()
    xfit, _ = make_batch(SurfaceMixtureV2(seed=8501), 256, a.length, 'cpu')
    activations = []
    def collect(_, inputs):
        activations.append(inputs[0].detach())
    handle = bond.register_forward_pre_hook(collect)
    with torch.no_grad():
        for x in xfit.split(32):
            model(x)
    handle.remove()
    z = torch.cat(activations).reshape(-1, model.config.n_embd).double()
    _, u = torch.linalg.eigh(z.T@z)
    activation_basis = torch.flip(u, (1,)).float()
    rng = np.random.default_rng(6511)
    random_basis = torch.from_numpy(np.linalg.qr(
        rng.normal(size=(model.config.n_embd, model.config.n_embd)))[0]).float()
    rows = []
    for width in [int(x) for x in a.widths.split(',')]:
        bases = {'weight_svd': weight_basis[:, :width],
                 'activation_pca': activation_basis[:, :width],
                 'random': random_basis[:, :width]}
        for kind in KINDS:
            bases[f'edit_{kind}'] = normalized_svd_basis(fit_by_kind[kind], width)
        for basis_name, basis in bases.items():
            rows.append(dict(width=width, basis=basis_name,
                             test={kind: energy(test_by_kind[kind], basis)
                                   for kind in KINDS}))
    result = dict(config=vars(a), fit_frames=fit_frames,
                  test_frames=test_frames,
                  fit_counts={kind: len(rows) for kind, rows in fit_by_kind.items()},
                  test_counts={kind: len(rows) for kind, rows in test_by_kind.items()},
                  rows=rows,
                  caveat='These are pre-output-projection activation differences at '
                         'the final position. Energy capture is a local geometry '
                         'test, not causal mediation or recovery of a complete rule. '
                         'The fit and test use different streams and disjoint frames.')
    Path(a.output).write_text(json.dumps(result, indent=2)+'\n')
    for row in rows:
        if row['width'] == 8:
            print(row['basis'], {k: round(v['mean'], 3)
                                 for k, v in row['test'].items()})


if __name__ == '__main__':
    main()

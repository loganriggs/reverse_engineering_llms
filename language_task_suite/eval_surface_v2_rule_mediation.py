"""Causal output test for rule-edit bases fitted on disjoint frames."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from eval_surface_bond_gauges import bases_for
from eval_surface_mixture_edits import predict
from eval_surface_mixture_v2_edits import build_pairs
from eval_surface_v2_rule_subspaces import (FIT_FRAMES, KINDS, TEST_FRAMES,
                                            normalized_svd_basis, response_deltas)
from eval_surface_v2_sparse_rotation import quartimax, unit_rows
from surface_mixture_v2 import SurfaceMixtureV2
from train_surface_mixture import make_batch


@torch.no_grad()
def predict_patched(model, bond, prefixes, deltas):
    out = torch.empty((len(prefixes), model.config.vocab_size),
                      dtype=torch.float32)
    groups = defaultdict(list)
    for i, prefix in enumerate(prefixes):
        groups[len(prefix)].append(i)
    current = [None]

    def patch(_, inputs):
        x = inputs[0].clone()
        x[:, -1] += current[0]
        return (x,)

    handle = bond.register_forward_pre_hook(patch)
    try:
        for indices in groups.values():
            for start in range(0, len(indices), 64):
                sub = indices[start:start+64]
                current[0] = deltas[sub]
                x = torch.tensor([prefixes[i] for i in sub], dtype=torch.long)
                out[sub] = F.softmax(model(x)[0][:, -1].float(), -1)
    finally:
        handle.remove()
    return out


def mean_l1(a, b):
    return float((a-b).abs().sum(-1).mean())


def mean_tv(a, b):
    return mean_l1(a, b)/2


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dir', default='language_task_suite/surface_mixture_v2_soft_models')
    p.add_argument('--layer', type=int, default=0)
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--nfit', type=int, default=256)
    p.add_argument('--ntest', type=int, default=256)
    p.add_argument('--length', type=int, default=72)
    p.add_argument('--width', type=int, default=8)
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
    learned = {kind: normalized_svd_basis(fit_by_kind[kind], a.width)
               for kind in KINDS}
    xfit, _ = make_batch(SurfaceMixtureV2(seed=8501), 256, a.length, 'cpu')
    captured = []
    def collect(_, inputs):
        captured.append(inputs[0].detach())
    handle = bond.register_forward_pre_hook(collect)
    with torch.no_grad():
        for x in xfit.split(32):
            model(x)
    handle.remove()
    generic = bases_for(torch.cat(captured), bond.weight.detach())
    prior32 = generic['weight_svd'][:, :32]
    valid_fit = [i for i, pair in enumerate(fit) if pair['kind'] in KINDS]
    unlabeled = fit_delta[valid_fit]@prior32
    response_rotation, _ = quartimax(unlabeled)
    activation_rows = torch.cat(captured).reshape(-1, model.config.n_embd)@prior32
    activation_rows = activation_rows.double()-activation_rows.double().mean(0)
    usage_rotation, _ = quartimax(activation_rows[:4096])
    sparse_axis_bases = {}
    for name, rotation in [('raw_weight32', torch.eye(32)),
                           ('response_quartimax', response_rotation),
                           ('usage_quartimax', usage_rotation)]:
        full_basis = prior32@rotation
        by_kind = {}
        for kind in KINDS:
            z = unit_rows(fit_by_kind[kind]@full_basis)
            axes = z.square().mean(0).topk(a.width).indices
            by_kind[kind] = full_basis[:, axes]
        sparse_axis_bases[name] = by_kind
    constrained = {}
    for prior_width in (16, 32):
        prior = generic['weight_svd'][:, :prior_width]
        constrained[prior_width] = {
            kind: prior@normalized_svd_basis(
                fit_by_kind[kind]@prior, a.width)
            for kind in KINDS}
    original = [pair['original'] for pair in test]
    edited = [pair['edited'] for pair in test]
    p0 = torch.from_numpy(predict(model, original, model.config.vocab_size))
    p1 = torch.from_numpy(predict(model, edited, model.config.vocab_size))
    full = predict_patched(model, bond, original, test_delta)
    oracle_effects = torch.zeros_like(p0)
    token2id = SurfaceMixtureV2().token2id
    for i, pair in enumerate(test):
        for token, mass in pair['q_after'].items():
            oracle_effects[i, token2id[token]] += mass
        for token, mass in pair['q_before'].items():
            oracle_effects[i, token2id[token]] -= mass
    results = {}
    for kind in KINDS:
        indices = [i for i, pair in enumerate(test) if pair['kind'] == kind]
        d = test_delta[indices]
        bases = {f'edit_{name}': basis for name, basis in learned.items()}
        for prior_width, by_kind in constrained.items():
            bases[f'weight{prior_width}_then_edit_{kind}'] = by_kind[kind]
        for name, by_kind in sparse_axis_bases.items():
            bases[f'{name}_select_{kind}'] = by_kind[kind]
        bases.update({'weight_svd': generic['weight_svd'][:, :a.width],
                 'activation_pca': generic['activation_pca'][:, :a.width],
                 'random': generic['random'][:, :a.width]})
        target_effect = p1[indices]-p0[indices]
        full_effect = full[indices]-p0[indices]
        oracle_effect = oracle_effects[indices]
        row = dict(n=len(indices), model_effect_tv=mean_tv(p1[indices], p0[indices]),
                   full_patch_tv=mean_tv(full[indices], p0[indices]),
                   full_patch_to_model_l1=mean_l1(full_effect, target_effect),
                   model_to_oracle_l1=mean_l1(target_effect, oracle_effect),
                   full_patch_to_oracle_l1=mean_l1(full_effect, oracle_effect),
                   bases={})
        prefixes = [original[i] for i in indices]
        for name, basis in bases.items():
            projected = (d@basis)@basis.T
            pp = predict_patched(model, bond, prefixes, projected)
            effect = pp-p0[indices]
            row['bases'][name] = dict(
                patch_tv=mean_tv(pp, p0[indices]),
                patch_to_full_l1=mean_l1(effect, full_effect),
                patch_to_model_l1=mean_l1(effect, target_effect),
                patch_to_oracle_l1=mean_l1(effect, oracle_effect))
        results[kind] = row
    output = dict(config=vars(a), fit_frames=fit_frames,
                  test_frames=test_frames, results=results,
                  caveat='Only the final-position pre-output-projection vector at '
                         'one bond is patched. Full local patch may not reproduce '
                         'the entire edited-model effect because residual paths, '
                         'earlier positions and other layers also change. Compare '
                         'projected patches with full local patches first; this '
                         'does not establish a complete or unique rule circuit.')
    Path(a.output).write_text(json.dumps(output, indent=2)+'\n')
    for kind, row in results.items():
        print(kind, 'model TV', round(row['model_effect_tv'], 3),
              'full TV', round(row['full_patch_tv'], 3),
              'edit-to-full L1', round(row['bases'][f'edit_{kind}']['patch_to_full_l1'], 3),
              'weight-to-full L1', round(row['bases']['weight_svd']['patch_to_full_l1'], 3))


if __name__ == '__main__':
    main()

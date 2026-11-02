"""Can a compact final-bond induction basis transfer across source distance?

Fit short-history induction responses on one set of sentence frames; test
long-history effects on disjoint frames and streams. A full local patch gates
any claim that a projected basis captures a causal model effect.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

from eval_surface_mixture_edits import predict
from eval_surface_mixture_v2_edits import build_pairs
from eval_surface_v2_rule_mediation import (mean_l1, mean_tv,
                                            predict_patched)
from eval_surface_v2_rule_subspaces import (FIT_FRAMES, TEST_FRAMES,
                                            normalized_svd_basis,
                                            response_deltas)
from eval_surface_v2_sparse_rotation import quartimax, unit_rows
from surface_mixture_v2 import SurfaceMixtureV2
from train_surface_mixture import make_batch


def subset(pairs, frames, low, high):
    return [pair for pair in pairs
            if pair['kind'] == 'induction' and pair['frame'] in frames
            and low <= pair['source_distance'] <= high]


def oracle_changes(pairs, vocab):
    out = torch.zeros(len(pairs), len(vocab), dtype=torch.float32)
    for i, pair in enumerate(pairs):
        for token, mass in pair['q_after'].items():
            out[i, vocab[token]] += mass
        for token, mass in pair['q_before'].items():
            out[i, vocab[token]] -= mass
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dir', default='language_task_suite/surface_mixture_v2_repeat01_return03_soft_models')
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--layer', type=int, default=1)
    p.add_argument('--width', type=int, default=8)
    p.add_argument('--prior-width', type=int, default=32)
    p.add_argument('--nfit', type=int, default=1024)
    p.add_argument('--ntest', type=int, default=1024)
    p.add_argument('--length', type=int, default=72)
    p.add_argument('--short-max', type=int, default=24)
    p.add_argument('--subject-repeat-mass', type=float, default=.1)
    p.add_argument('--subject-return-mass', type=float, default=.3)
    p.add_argument('--fit-seed', type=int, default=9901)
    p.add_argument('--test-seed', type=int, default=8701)
    p.add_argument('--reverse-frames', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if not 0 < a.width <= a.prior_width:
        p.error('width must lie in [1, prior-width]')
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
    dgp = dict(subject_repeat_mass=a.subject_repeat_mass,
               subject_return_mass=a.subject_return_mass)
    fit_frames = TEST_FRAMES if a.reverse_frames else FIT_FRAMES
    test_frames = FIT_FRAMES if a.reverse_frames else TEST_FRAMES
    fit_all = build_pairs(SurfaceMixtureV2(seed=a.fit_seed, **dgp),
                          a.nfit, a.length)
    test_all = build_pairs(SurfaceMixtureV2(seed=a.test_seed, **dgp),
                           a.ntest, a.length)
    fit_short = subset(fit_all, fit_frames, 1, a.short_max)
    fit_long = subset(fit_all, fit_frames, a.short_max+1, 10**9)
    test_long = subset(test_all, test_frames, a.short_max+1, 10**9)
    if min(len(fit_short), len(fit_long), len(test_long)) < a.prior_width:
        raise ValueError('Insufficient induction edits for requested width')

    short_delta = response_deltas(model, bond, fit_short)
    long_delta = response_deltas(model, bond, fit_long)
    test_delta = response_deltas(model, bond, test_long)
    _, _, vh = torch.linalg.svd(bond.weight.detach().double(),
                                full_matrices=False)
    weight = vh.T.float()
    prior = weight[:, :a.prior_width]
    short_basis = normalized_svd_basis(short_delta, a.width)
    long_basis = normalized_svd_basis(long_delta, a.width)
    constrained = prior@normalized_svd_basis(short_delta@prior, a.width)
    rotation, objective = quartimax(short_delta@prior)
    rotated = prior@rotation
    fit_energy = unit_rows(short_delta@rotated).square().mean(0)
    sparse = rotated[:, fit_energy.topk(a.width).indices]

    xfit, _ = make_batch(SurfaceMixtureV2(seed=8501, **dgp), 256,
                         a.length, 'cpu')
    captured = []
    handle = bond.register_forward_pre_hook(
        lambda _, inputs: captured.append(inputs[0].detach()))
    with torch.no_grad():
        for x in xfit.split(32):
            model(x)
    handle.remove()
    activations = torch.cat(captured).reshape(-1, model.config.n_embd)
    centered = activations.double()-activations.double().mean(0)
    _, vectors = torch.linalg.eigh(centered.T@centered)
    activation_basis = vectors.flip(1)[:, :a.width].float()

    bases = dict(weight_svd=weight[:, :a.width],
                 activation_pca=activation_basis,
                 short_edit_svd=short_basis,
                 weight_prior_then_short_edit=constrained,
                 weight_prior_short_quartimax=sparse,
                 long_edit_svd_control=long_basis)
    rng = torch.Generator().manual_seed(7141)
    bases['random'] = torch.linalg.qr(torch.randn(
        model.config.n_embd, model.config.n_embd,
        generator=rng))[0][:, :a.width]
    original = [pair['original'] for pair in test_long]
    edited = [pair['edited'] for pair in test_long]
    p0 = torch.from_numpy(predict(model, original, model.config.vocab_size))
    p1 = torch.from_numpy(predict(model, edited, model.config.vocab_size))
    full = predict_patched(model, bond, original, test_delta)
    oracle = oracle_changes(test_long, SurfaceMixtureV2().token2id)
    projected = {name: predict_patched(
        model, bond, original, (test_delta@basis)@basis.T)
        for name, basis in bases.items()}

    groups = {'25_to_48': [i for i, pair in enumerate(test_long)
                           if pair['source_distance'] <= 48],
              'over_48': [i for i, pair in enumerate(test_long)
                          if pair['source_distance'] > 48],
              'all_long': list(range(len(test_long)))}
    results = {}
    for group, idx in groups.items():
        if not idx:
            continue
        target = p1[idx]-p0[idx]
        full_effect = full[idx]-p0[idx]
        oracle_effect = oracle[idx]
        results[group] = dict(
            n=len(idx), model_effect_tv=mean_tv(p1[idx], p0[idx]),
            full_patch_tv=mean_tv(full[idx], p0[idx]),
            full_patch_to_model_l1=mean_l1(full_effect, target),
            model_to_oracle_l1=mean_l1(target, oracle_effect),
            full_patch_to_oracle_l1=mean_l1(full_effect, oracle_effect),
            bases={name: dict(
                patch_tv=mean_tv(pp[idx], p0[idx]),
                patch_to_full_l1=mean_l1(pp[idx]-p0[idx], full_effect),
                patch_to_model_l1=mean_l1(pp[idx]-p0[idx], target),
                patch_to_oracle_l1=mean_l1(pp[idx]-p0[idx], oracle_effect))
                for name, pp in projected.items()})
    output = dict(config={**vars(a), 'output': str(a.output)},
                  fit_frames=fit_frames, test_frames=test_frames,
                  fit_short_n=len(fit_short), fit_long_n=len(fit_long),
                  test_long_n=len(test_long),
                  short_quartimax_objective=objective,
                  results=results,
                  caveat='Short/long edit bases see model activation responses, '
                         'not oracle labels; the long-fit basis is an informed '
                         'control, not a guaranteed upper bound. Fit and test '
                         'frames and streams '
                         'are disjoint. A full single-position bond patch can '
                         'miss residual and earlier-position effects. Inspect '
                         'full-patch mediation before reading projected results.')
    a.output.write_text(json.dumps(output, indent=2)+'\n')
    for group, row in results.items():
        print(group, 'n', row['n'], 'model/full TV',
              round(row['model_effect_tv'], 3),
              round(row['full_patch_tv'], 3),
              'full-to-model L1', round(row['full_patch_to_model_l1'], 3),
              'short-edit-to-full', round(row['bases']['short_edit_svd']['patch_to_full_l1'], 3),
              'weight-to-full', round(row['bases']['weight_svd']['patch_to_full_l1'], 3),
              flush=True)


if __name__ == '__main__':
    main()

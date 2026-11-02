"""Fit unlabeled local featurizers, then test an omitted rule family causally.

The language model sees every rule. Featurizer fitting uses native attention-bond
activations from lexical and induction edit contexts only; edit labels are used
to select/balance the fitting panel, never supplied to the featurizer.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from eval_surface_mixture_edits import predict
from eval_surface_mixture_v2_edits import build_pairs
from eval_surface_v2_rule_mediation import mean_l1, mean_tv, predict_patched
from eval_surface_v2_rule_subspaces import FIT_FRAMES, KINDS, TEST_FRAMES, capture
from surface_mixture_v2 import SurfaceMixtureV2, VERBS
from surface_mixture_v2_varied import SurfaceMixtureV2Varied


class ScalarTopK(nn.Module):
    def __init__(self, dimension, atoms, k):
        super().__init__()
        self.k = k
        self.dictionary = nn.Parameter(torch.randn(dimension, atoms))
        self.encoder = nn.Linear(dimension, atoms)
        self.bias = nn.Parameter(torch.zeros(dimension))

    def forward(self, x):
        a = self.encoder(x)
        support = torch.zeros_like(a).scatter_(-1, a.abs().topk(self.k, -1).indices, 1)
        code = a * support
        decoder = F.normalize(self.dictionary, dim=0)
        return dict(reconstruction=code@decoder.T+self.bias,
                    support=support)


def balanced_pairs(seed, nstreams, length, frames, kinds, rng_seed,
                   generator_class=SurfaceMixtureV2, generator_kwargs=None):
    pairs = [pair for pair in build_pairs(generator_class(
        seed=seed, **(generator_kwargs or {})),
                                         nstreams, length)
             if pair['frame'] in frames and pair['kind'] in kinds]
    n = min(sum(pair['kind'] == kind for pair in pairs) for kind in kinds)
    generator = torch.Generator().manual_seed(rng_seed)
    chosen = []
    for kind in kinds:
        rows = [pair for pair in pairs if pair['kind'] == kind]
        indices = torch.randperm(len(rows), generator=generator)[:n].tolist()
        chosen.extend(rows[i] for i in indices)
    return chosen


def activations(model, bond, pairs):
    before = capture(model, bond, [pair['original'] for pair in pairs])
    after = capture(model, bond, [pair['edited'] for pair in pairs])
    return before, after


def initialize(model, directions):
    with torch.no_grad():
        model.dictionary.copy_(directions)
        encoder = (model.coordinate_encoder if hasattr(model, 'coordinate_encoder')
                   else model.encoder)
        encoder.weight.copy_(directions.T)
        encoder.bias.zero_()
        model.bias.zero_()


def fit(model, train, val, steps, seed, fixed_membership=None,
        coverage_weight=0., change_weight=0., train_pairs=None,
        val_pairs=None, output_contrast=None, output_contrast_weight=0.):
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=.002, weight_decay=.0001)
    generator = torch.Generator().manual_seed(seed)
    best = float('inf')
    chosen_state = None
    records = []
    for step in range(1, steps+1):
        if change_weight or output_contrast_weight:
            ix = torch.randint(len(train_pairs[0]), (128,), generator=generator)
            x = torch.cat((train_pairs[0][ix], train_pairs[1][ix]))
        else:
            ix = torch.randint(len(train), (256,), generator=generator)
            x = train[ix]
        model.train()
        result = (model(x, membership=fixed_membership)
                  if fixed_membership is not None else model(x))
        reconstruction_loss = F.mse_loss(result['reconstruction'], x)
        loss = reconstruction_loss
        if change_weight or output_contrast_weight:
            r0, r1 = result['reconstruction'].chunk(2)
            target_delta = train_pairs[1][ix]-train_pairs[0][ix]
            reconstructed_delta = r1-r0
            if change_weight:
                loss = loss + change_weight*F.mse_loss(
                    reconstructed_delta, target_delta)
            if output_contrast_weight:
                loss = loss + output_contrast_weight*F.mse_loss(
                    reconstructed_delta@output_contrast.T,
                    target_delta@output_contrast.T)
        if coverage_weight:
            covered = 1 - (1 - result['membership']).prod(0)
            # Sixty-four links may still overlap; require 48 distinct atoms.
            loss = loss + coverage_weight * F.relu(
                .75-covered.mean()).square()
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss at step {step}')
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
        optimizer.step()
        if step == 1 or step % 200 == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                result = (model(val, membership=fixed_membership)
                          if fixed_membership is not None else model(val))
                score = float(F.mse_loss(result['reconstruction'], val))
                active = float(result['support'].sum(-1).float().mean())
                val_change = None
                val_output_contrast = None
                if change_weight or output_contrast_weight:
                    v0, v1 = val_pairs
                    pair_result = (model(torch.cat((v0, v1)),
                                         membership=fixed_membership)
                                   if fixed_membership is not None
                                   else model(torch.cat((v0, v1))))
                    pr0, pr1 = pair_result['reconstruction'].chunk(2)
                    if change_weight:
                        val_change = float(F.mse_loss(pr1-pr0, v1-v0))
                    if output_contrast_weight:
                        val_output_contrast = float(F.mse_loss(
                            (pr1-pr0)@output_contrast.T,
                            (v1-v0)@output_contrast.T))
                val_coverage = 0.
                if coverage_weight:
                    covered = 1 - (1 - result['membership']).prod(0)
                    val_coverage = float(F.relu(.75-covered.mean()).square())
            objective = (score + change_weight*(val_change or 0.)
                             + output_contrast_weight*(val_output_contrast or 0.)
                             + coverage_weight*val_coverage)
            records.append(dict(step=step, train_objective=float(loss.detach()),
                                val_mse=score, val_change_mse=val_change,
                                val_output_contrast_mse=val_output_contrast,
                                val_coverage_penalty=val_coverage,
                                val_objective=objective, active_atoms=active))
            if objective < best:
                best = objective
                chosen_state = {key: value.detach().clone()
                                for key, value in model.state_dict().items()}
    model.load_state_dict(chosen_state)
    model.eval()
    return records


@torch.no_grad()
def reconstruct(name, model, x, fixed_membership, pca):
    if name == 'pca8':
        return (x@pca)@pca.T, None
    result = (model(x, membership=fixed_membership)
              if fixed_membership is not None else model(x))
    return result['reconstruction'], result['support']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--feature-source', default='/home/loganriggs/Coding/neuron_circuits/scripts')
    p.add_argument('--model-dir', type=Path,
                   default=Path('language_task_suite/surface_mixture_v2_soft_models'))
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--layer', type=int, default=1)
    p.add_argument('--length', type=int, default=72)
    p.add_argument('--nfit', type=int, default=1024)
    p.add_argument('--nval', type=int, default=256)
    p.add_argument('--ntest', type=int, default=256)
    p.add_argument('--test-seed', type=int, default=20261113)
    p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--coverage-weight', type=float, default=0.)
    p.add_argument('--change-weight', type=float, default=0.)
    p.add_argument('--output-contrast-weight', type=float, default=0.,
                   help='Supervise paired changes in known verb-output contrasts')
    p.add_argument('--groups', type=int, default=16)
    p.add_argument('--group-budget', type=int, default=2)
    p.add_argument('--membership-budget', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--reverse-frames', action='store_true')
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    sys.path.insert(0, a.source)
    sys.path.insert(0, a.feature_source)
    from feature_structures.models import SharedGroups
    from model import BilinearGPT, Config

    saved = json.loads((a.model_dir/'results.json').read_text())
    generator_name = saved.get('config', {}).get('generator', 'v2')
    generator_class = {'v2': SurfaceMixtureV2,
                       'v2_varied': SurfaceMixtureV2Varied}[generator_name]
    generator_kwargs = dict(saved.get('generator_kwargs', {}))
    row = next(row for row in saved['records'] if row['depth'] == a.depth)
    language_model = BilinearGPT(Config(**row['model_config']))
    language_model.load_state_dict(torch.load(
        a.model_dir/f'model_depth{a.depth}.pt', map_location='cpu',
        weights_only=True))
    language_model.eval()
    bond = language_model.blocks[a.layer].attn.out
    forward_fit = ('lead', 'at', 'when') if generator_name == 'v2_varied' else FIT_FRAMES
    forward_test = ('subject', 'in_place', 'during') if generator_name == 'v2_varied' else TEST_FRAMES
    fit_frames = forward_test if a.reverse_frames else forward_fit
    test_frames = forward_fit if a.reverse_frames else forward_test
    kinds = (KINDS + ('cue', 'nuisance_determiner')
             if generator_name == 'v2_varied' else KINDS)
    fit_kinds = ('lexical', 'induction')
    fit_pairs = balanced_pairs(9901+a.seed, a.nfit, a.length,
                               fit_frames, fit_kinds, 6211+a.seed,
                               generator_class, generator_kwargs)
    val_pairs = balanced_pairs(9902+a.seed, a.nval, a.length,
                               fit_frames, fit_kinds, 6212+a.seed,
                               generator_class, generator_kwargs)
    test_pairs = [pair for pair in build_pairs(
        generator_class(seed=a.test_seed, **generator_kwargs),
        a.ntest, a.length)
        if pair['frame'] in test_frames]
    zfit0, zfit1 = activations(language_model, bond, fit_pairs)
    zval0, zval1 = activations(language_model, bond, val_pairs)
    ztest0, ztest1 = activations(language_model, bond, test_pairs)
    native_train = torch.cat((zfit0, zfit1))
    native_val = torch.cat((zval0, zval1))
    center = native_train.mean(0)
    scale = (native_train-center).square().mean().sqrt().clamp_min(1e-6)
    train = (native_train-center)/scale
    val = (native_val-center)/scale
    train_pair_activations = ((zfit0-center)/scale, (zfit1-center)/scale)
    val_pair_activations = ((zval0-center)/scale, (zval1-center)/scale)
    output_contrast = None
    if a.output_contrast_weight:
        output = language_model.lm_head.weight.detach()
        if hasattr(language_model, 'final_norm'):
            output = output*language_model.final_norm.weight.detach()[None]
        vocab = generator_class(**generator_kwargs).token2id
        verb_rows = output[[vocab[token] for token in VERBS]]
        output_contrast = (verb_rows-verb_rows.mean(0))@bond.weight.detach()
        # Unit mean row norm makes an isotropic unit-variance activation delta
        # have output-contrast variance near one. This is a local linear map.
        output_contrast = (output_contrast /
                           output_contrast.square().sum(1).mean().sqrt())
    x0 = (ztest0-center)/scale
    x1 = (ztest1-center)/scale
    _, _, vh = torch.linalg.svd(train.double(), full_matrices=False)
    directions = vh.T.float()
    torch.manual_seed(7011+a.seed)
    assert a.groups*a.membership_budget == 64, 'disjoint control needs 64 memberships'
    assert a.group_budget*a.membership_budget == 8, 'match scalar Top-8 maximum support'
    learned = SharedGroups(64, 64, a.groups, a.group_budget,
                           a.membership_budget)
    initialize(learned, directions)
    disjoint = SharedGroups(64, 64, a.groups, a.group_budget,
                            a.membership_budget)
    disjoint.load_state_dict(learned.state_dict())
    disjoint.membership_logits.requires_grad_(False)
    membership = torch.zeros(a.groups, 64)
    for atom in range(64):
        membership[atom % a.groups, atom] = 1
    scalar6 = ScalarTopK(64, 64, 6)
    scalar8 = ScalarTopK(64, 64, 8)
    for scalar in (scalar6, scalar8):
        initialize(scalar, directions)
    models = {'shared_groups': learned, 'disjoint_groups': disjoint,
              'scalar_top6': scalar6, 'scalar_top8': scalar8}
    histories = {}
    for name, model in models.items():
        fixed = membership if name == 'disjoint_groups' else None
        histories[name] = fit(model, train, val, a.steps, 7300+a.seed, fixed,
                              a.coverage_weight if name == 'shared_groups' else 0.,
                              a.change_weight, train_pair_activations,
                              val_pair_activations, output_contrast,
                              a.output_contrast_weight)
        print(name, 'best objective', min(row['val_objective'] for row in histories[name]),
              'active', histories[name][-1]['active_atoms'], flush=True)

    original = [pair['original'] for pair in test_pairs]
    edited = [pair['edited'] for pair in test_pairs]
    p0 = torch.from_numpy(predict(language_model, original,
                                  language_model.config.vocab_size))
    p1 = torch.from_numpy(predict(language_model, edited,
                                  language_model.config.vocab_size))
    full_delta = ztest1-ztest0
    full = predict_patched(language_model, bond, original, full_delta)
    oracle = torch.zeros_like(p0)
    token2id = generator_class(**generator_kwargs).token2id
    for i, pair in enumerate(test_pairs):
        for token, mass in pair['q_after'].items():
            oracle[i, token2id[token]] += mass
        for token, mass in pair['q_before'].items():
            oracle[i, token2id[token]] -= mass
    rows = {}
    methods = {'pca8': None, **models}
    for name, model in methods.items():
        fixed = membership if name == 'disjoint_groups' else None
        r0, s0 = reconstruct(name, model, x0, fixed, directions[:, :8])
        r1, s1 = reconstruct(name, model, x1, fixed, directions[:, :8])
        reconstructed_delta = (r1-r0)*scale
        patched = predict_patched(language_model, bond, original,
                                  reconstructed_delta)
        result = dict(test_reconstruction_mse=float((
                          torch.cat((r0-x0, r1-x1)).square().mean())),
                      test_change_mse=float(F.mse_loss(r1-r0, x1-x0)),
                      active_atoms=(8. if s0 is None else float(torch.cat((
                          s0, s1)).sum(-1).float().mean())),
                      parameters=(0 if model is None else sum(
                          p.numel() for p in model.parameters()
                          if p.requires_grad)),
                      by_kind={})
        for kind in kinds:
            ix = [i for i, pair in enumerate(test_pairs)
                  if pair['kind'] == kind]
            effect = patched[ix]-p0[ix]
            full_effect = full[ix]-p0[ix]
            model_effect = p1[ix]-p0[ix]
            result['by_kind'][kind] = dict(
                n=len(ix), patch_tv=mean_tv(patched[ix], p0[ix]),
                patch_to_full_l1=mean_l1(effect, full_effect),
                patch_to_model_l1=mean_l1(effect, model_effect),
                patch_to_oracle_l1=mean_l1(effect, oracle[ix]),
                full_patch_to_model_l1=mean_l1(full_effect, model_effect),
                model_to_oracle_l1=mean_l1(model_effect, oracle[ix]))
        rows[name] = result
        print(name, 'place holdout L1', round(
            result['by_kind']['place_skip']['patch_to_full_l1'], 4), flush=True)
    with torch.no_grad():
        train_groups = learned(train)
        group_membership = train_groups['membership'].detach()
        assert torch.all(group_membership.sum(-1) == a.membership_budget)
        assert torch.all(train_groups['seed_gates'].sum(-1) == a.group_budget)
        assert torch.all(train_groups['support'].sum(-1) <= 8)
        assert torch.all(membership.sum(-1) == a.membership_budget)
        shared_unique_atoms = int((group_membership.sum(0) > 0).sum())
        shared_reused_atoms = int((group_membership.sum(0) > 1).sum())
        test_group_before = learned(x0)['gates']
        test_group_after = learned(x1)['gates']
        gate_diagnostics = dict(
            train_group_usage=train_groups['gates'].mean(0).tolist(),
            membership_degree=group_membership.sum(0).tolist(),
            by_kind={})
        for kind in kinds:
            ix = [i for i, pair in enumerate(test_pairs)
                  if pair['kind'] == kind]
            change = (test_group_after[ix]-test_group_before[ix]).abs()
            gate_diagnostics['by_kind'][kind] = dict(
                n=len(ix), mean_gate_changes=float(change.sum(-1).mean()/2),
                changed_group_frequency=change.mean(0).tolist())
    a.output_dir.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        torch.save(model.state_dict(), a.output_dir/f'{name}.pt')
    output = dict(config={**vars(a), 'source': str(a.source),
                          'feature_source': str(a.feature_source),
                          'model_dir': str(a.model_dir),
                          'output_dir': str(a.output_dir)},
                  generator_name=generator_name,
                  generator_kwargs=generator_kwargs,
                  fit_frames=fit_frames, test_frames=test_frames,
                  fit_kinds=fit_kinds,
                  fit_pairs=len(fit_pairs), val_pairs=len(val_pairs),
                  train_change_mse=float(F.mse_loss(
                      train_pair_activations[1], train_pair_activations[0])),
                  test_counts={kind: sum(pair['kind'] == kind
                                         for pair in test_pairs)
                               for kind in kinds},
                  scale=float(scale), histories=histories, results=rows,
                  shared_membership=dict(unique_atoms=shared_unique_atoms,
                                         reused_atoms=shared_reused_atoms),
                  gate_diagnostics=gate_diagnostics,
                  caveat='Featurizer fit uses native bond activations '
                         'from balanced lexical/induction edit contexts. '
                         'A nonzero output-contrast weight additionally gives '
                         'the objective known verb-output labels and paired edits; '
                         'it is supervised and locally linearized. '
                         'Previous-place is held out from featurizer fit, not '
                         'from language-model training. Reconstruction and '
                         'local patch fidelity do not establish task discovery. '
                         'Nonlinear encoders reconstruct endpoints separately.')
    (a.output_dir/'results.json').write_text(json.dumps(output, indent=2)+'\n')


if __name__ == '__main__':
    main()

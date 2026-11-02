"""Whole-genre holdout for scalar and shared-group attention-bond featurizers.

Fit only on Reports/Policy activations; test FAQ/Guide prediction and valid
causal edits. The language model itself was trained on all four genres.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from eval_multi_genre_bond_gauges import group_scores
from eval_multi_genre_edits import build_pairs
from eval_surface_mixture_edits import predict, score
from multi_genre_language import GENRES, MultiGenreLanguage
from train_multi_genre import make_batch, make_eval
from train_surface_v2_shared_groups import ScalarTopK, fit, initialize


@torch.no_grad()
def capture_bond(model, bond, x):
    chunks = []
    def collect(_, inputs):
        chunks.append(inputs[0].detach().cpu())
    hook = bond.register_forward_pre_hook(collect)
    try:
        for batch in x.split(32):
            model(batch)
    finally:
        hook.remove()
    return torch.cat(chunks).reshape(-1, model.config.n_embd)


@torch.no_grad()
def capture_final_context(model, bond, x):
    residuals, bonds, logits = [], [], []
    block = model.blocks[-1]
    def residual_hook(_, inputs):
        residuals.append(inputs[0].detach().cpu())
    def bond_hook(_, inputs):
        bonds.append(inputs[0].detach().cpu())
    hooks = [block.register_forward_pre_hook(residual_hook),
             bond.register_forward_pre_hook(bond_hook)]
    try:
        for batch in x.split(32):
            logits.append(model(batch)[0].detach().cpu())
    finally:
        for hook in hooks:
            hook.remove()
    residual = torch.cat(residuals).reshape(-1, model.config.n_embd)
    z = torch.cat(bonds).reshape(-1, model.config.n_embd)
    expected = torch.cat(logits).reshape(-1, model.config.vocab_size)
    replay = model.lm_head(model.final_norm(residual + bond(z)))
    error = float((replay-expected).abs().max())
    if error > 2e-5:
        raise AssertionError(f'Final-bond replay error {error}')
    return dict(residual=residual, bond=z, logits=expected,
                replay_max_logit_error=error)


def fit_output_aware(feature, train, val, train_context, val_context,
                     center, scale, language_model, bond, steps, seed,
                     output_kl_weight, fixed_membership=None,
                     coverage_weight=0.):
    optimizer = torch.optim.AdamW(
        (p for p in feature.parameters() if p.requires_grad),
        lr=.002, weight_decay=.0001)
    rng = torch.Generator().manual_seed(seed)
    target_logp = F.log_softmax(train_context['logits'], -1)
    target_prob = target_logp.exp()
    val_logp = F.log_softmax(val_context['logits'], -1)
    val_prob = val_logp.exp()
    best = float('inf')
    best_state = None
    records = []

    def objective(x, residual, logp, prob):
        result = (feature(x, membership=fixed_membership)
                  if fixed_membership is not None else feature(x))
        reconstruction = result['reconstruction']
        bond_reconstruction = reconstruction*scale+center
        output = language_model.lm_head(language_model.final_norm(
            residual + bond(bond_reconstruction)))
        kl = (prob*(logp-F.log_softmax(output, -1))).sum(-1).mean()
        mse = F.mse_loss(reconstruction, x)
        penalty = 0.
        if coverage_weight:
            covered = 1 - (1-result['membership']).prod(0)
            penalty = F.relu(.75-covered.mean()).square()
        loss = mse + output_kl_weight*kl + coverage_weight*penalty
        return loss, mse, kl, result

    for step in range(1, steps+1):
        ix = torch.randint(len(train), (256,), generator=rng)
        feature.train()
        loss, mse, kl, result = objective(
            train[ix], train_context['residual'][ix],
            target_logp[ix], target_prob[ix])
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite output-aware loss at step {step}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(feature.parameters(), 10.)
        optimizer.step()
        if step == 1 or step % 200 == 0 or step == steps:
            feature.eval()
            with torch.no_grad():
                value, vmse, vkl, vresult = objective(
                    val, val_context['residual'], val_logp, val_prob)
                score = float(value)
            records.append(dict(step=step, train_objective=float(loss.detach()),
                                train_mse=float(mse.detach()),
                                train_output_kl=float(kl.detach()),
                                val_objective=score, val_mse=float(vmse),
                                val_output_kl=float(vkl),
                                active_atoms=float(vresult['support'].sum(-1).mean())))
            if score < best:
                best = score
                best_state = {key: value.detach().clone()
                              for key, value in feature.state_dict().items()}
    feature.load_state_dict(best_state)
    feature.eval()
    return records


def hook_for(model, center, scale, fixed_membership=None, basis=None, keep=16):
    def hook(_, inputs):
        z = inputs[0]
        if basis is not None:
            coordinates = z @ basis
            mask = torch.zeros_like(coordinates).scatter_(
                -1, coordinates.abs().topk(keep, -1).indices, 1.)
            return ((coordinates * mask) @ basis.T,)
        x = (z - center) / scale
        result = (model(x, membership=fixed_membership)
                  if fixed_membership is not None else model(x))
        return (result['reconstruction'] * scale + center,)
    return hook


@torch.no_grad()
def evaluate_candidate(language_model, bond, feature_model, center, scale,
                       membership, basis, keep, test, native_logits,
                       native_logp, pairs, originals, edits, gen):
    stats = {}
    squared_error = 0.
    squared_energy = 0.
    if feature_model is not None:
        feature_model.eval()
        total_active = 0.
        total_vectors = 0
        used = torch.zeros(64, dtype=torch.bool)
        for batch in test['x'].split(32):
            z = capture_bond(language_model, bond, batch)
            x = (z-center)/scale
            result = (feature_model(x, membership=membership)
                      if membership is not None else feature_model(x))
            support = result['support'].reshape(-1, 64)
            reconstructed = result['reconstruction']*scale+center
            squared_error += float((z-reconstructed).square().sum())
            squared_energy += float(z.square().sum())
            total_active += float(support.sum())
            total_vectors += len(support)
            used |= support.bool().any(0)
        stats['active_atoms'] = total_active/total_vectors
        stats['support_coverage'] = int(used.sum())
    else:
        stats['active_atoms'] = keep
        for batch in test['x'].split(32):
            z = capture_bond(language_model, bond, batch)
            coordinates = z @ basis
            mask = torch.zeros_like(coordinates).scatter_(
                -1, coordinates.abs().topk(keep, -1).indices, 1.)
            reconstructed = (coordinates*mask)@basis.T
            squared_error += float((z-reconstructed).square().sum())
            squared_energy += float(z.square().sum())
    stats['relative_bond_mse'] = squared_error/max(squared_energy, 1e-30)
    hook = bond.register_forward_pre_hook(hook_for(
        feature_model, center, scale, membership, basis, keep))
    try:
        logits = torch.cat([language_model(batch)[0].float()
                            for batch in test['x'].split(32)])
        before = predict(language_model, originals, len(gen.vocab))
        after = predict(language_model, edits, len(gen.vocab))
    finally:
        hook.remove()
    effect = score(pairs, before, after, gen)
    return dict(groups=group_scores(logits, test, native_logp),
                edits=effect,
                max_logit_difference=float((logits-native_logits).abs().max()),
                **stats)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--feature-source', default='/home/loganriggs/Coding/neuron_circuits/scripts')
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--depth', type=int, default=3)
    p.add_argument('--layer', type=int, default=2)
    p.add_argument('--nfit', type=int, default=512)
    p.add_argument('--nval', type=int, default=128)
    p.add_argument('--ntest', type=int, default=1024)
    p.add_argument('--nedit-streams', type=int, default=512)
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--length', type=int, default=71)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--coverage-weight', type=float, default=1.)
    p.add_argument('--output-kl-weight', type=float, default=0.,
                   help='Exact final-bond downstream teacher KL on fit genres')
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    sys.path.insert(0, a.source)
    sys.path.insert(0, a.feature_source)
    from model import BilinearGPT, Config
    from feature_structures.models import SharedGroups

    saved = json.loads((a.model_dir/'results.json').read_text())
    record = next(r for r in saved['records'] if r['depth'] == a.depth)
    language_model = BilinearGPT(Config(**record['model_config']))
    language_model.load_state_dict(torch.load(
        a.model_dir/f'model_depth{a.depth}.pt',
        map_location='cpu', weights_only=True))
    language_model.eval().requires_grad_(False)
    bond = language_model.blocks[a.layer].attn.out
    fit_weights = tuple(float(genre in ('Reports', 'Policy')) for genre in GENRES)
    train_x, _ = make_batch(MultiGenreLanguage(seed=98101+a.seed,
                                               genre_weights=fit_weights),
                            a.nfit, a.length, 'cpu')
    val_x, _ = make_batch(MultiGenreLanguage(seed=98102+a.seed,
                                             genre_weights=fit_weights),
                          a.nval, a.length, 'cpu')
    token_names = MultiGenreLanguage().id2token
    assert all(token_names[int(token)] in ('Reports', 'Policy')
               for token in torch.cat((train_x[:, 0], val_x[:, 0])))
    if a.output_kl_weight:
        if a.layer != a.depth-1 or language_model.blocks[-1].mlp is not None:
            raise ValueError('Exact output KL currently requires final attention-only block')
        train_context = capture_final_context(language_model, bond, train_x)
        val_context = capture_final_context(language_model, bond, val_x)
        native_train = train_context['bond']
        native_val = val_context['bond']
    else:
        train_context = val_context = None
        native_train = capture_bond(language_model, bond, train_x)
        native_val = capture_bond(language_model, bond, val_x)
    center = native_train.mean(0)
    scale = (native_train-center).square().mean().sqrt().clamp_min(1e-6)
    train = (native_train-center)/scale
    val = (native_val-center)/scale
    _, _, vh = torch.linalg.svd(train.double(), full_matrices=False)
    directions = vh.T.float()
    torch.manual_seed(98200+a.seed)
    shared = SharedGroups(64, 64, 16, 4, 4)
    initialize(shared, directions)
    disjoint = SharedGroups(64, 64, 16, 4, 4)
    disjoint.load_state_dict(shared.state_dict())
    disjoint.membership_logits.requires_grad_(False)
    fixed_membership = torch.zeros(16, 64)
    for atom in range(64):
        fixed_membership[atom % 16, atom] = 1.
    scalar8 = ScalarTopK(64, 64, 8)
    scalar16 = ScalarTopK(64, 64, 16)
    for scalar in (scalar8, scalar16):
        initialize(scalar, directions)
    models = {'scalar8': scalar8, 'scalar16': scalar16,
              'disjoint16': disjoint, 'shared16': shared}
    histories = {}
    for name, feature in models.items():
        membership = fixed_membership if name == 'disjoint16' else None
        if a.output_kl_weight:
            histories[name] = fit_output_aware(
                feature, train, val, train_context, val_context,
                center, scale, language_model, bond, a.steps, 98300+a.seed,
                a.output_kl_weight, membership,
                a.coverage_weight if name == 'shared16' else 0.)
        else:
            histories[name] = fit(
                feature, train, val, a.steps, 98300+a.seed, membership,
                a.coverage_weight if name == 'shared16' else 0.)
        print(name, 'fit', histories[name][-1], flush=True)

    test = make_eval(MultiGenreLanguage(seed=98103), a.ntest,
                     a.length, 'cpu')
    with torch.no_grad():
        native_logits = torch.cat([language_model(batch)[0].float()
                                   for batch in test['x'].split(32)])
    native_logp = F.log_softmax(native_logits, -1)
    gen = MultiGenreLanguage(seed=98104)
    all_pairs = build_pairs(gen, a.nedit_streams, a.length)
    kinds = {'faq_answer_copy', 'faq_place_recall',
             'policy_condition', 'policy_nuisance_person'}
    pairs = [row for row in all_pairs if row['kind'] in kinds]
    originals = [row['original'] for row in pairs]
    edits = [row['edited'] for row in pairs]
    assert any(row['kind'] == 'faq_place_recall' for row in pairs)
    native_before = predict(language_model, originals, len(gen.vocab))
    native_after = predict(language_model, edits, len(gen.vocab))
    native = dict(groups=group_scores(native_logits, test, native_logp),
                  edits=score(pairs, native_before, native_after, gen))
    _, _, weight_vh = torch.linalg.svd(bond.weight.detach().double(),
                                       full_matrices=False)
    candidates = {
        'physical16': (None, None, torch.eye(64), 16),
        'weight_svd16': (None, None, weight_vh.T.float(), 16),
        'scalar8': (scalar8, None, None, 8),
        'scalar16': (scalar16, None, None, 16),
        'disjoint16': (disjoint, fixed_membership, None, 16),
        'shared16': (shared, None, None, 16),
    }
    rows = {}
    for name, (feature, membership, basis, keep) in candidates.items():
        rows[name] = evaluate_candidate(
            language_model, bond, feature, center, scale, membership, basis,
            keep, test, native_logits, native_logp, pairs, originals, edits,
            gen)
        summary = rows[name]
        print(name,
              'faq_kl', summary['groups']['faq_place_recall']['model_to_truncated_kl'],
              'faq_edit', summary['edits']['faq_place_recall']['effect_l1_error'],
              'active', summary['active_atoms'], flush=True)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    result = dict(config={**vars(a), 'model_dir': str(a.model_dir),
                          'output_dir': str(a.output_dir)},
                  fit_genres=['Reports', 'Policy'],
                  heldout_genres=['FAQ', 'Guide'],
                  fit_vectors=len(train), val_vectors=len(val),
                  final_bond_replay_max_error=(train_context['replay_max_logit_error']
                                               if train_context else None),
                  edit_pair_counts={kind: sum(row['kind'] == kind for row in pairs)
                                    for kind in sorted(kinds)},
                  native=native, candidates=rows, histories=histories)
    (a.output_dir/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    for name, feature in models.items():
        torch.save(feature.state_dict(), a.output_dir/f'{name}.pt')


if __name__ == '__main__':
    main()

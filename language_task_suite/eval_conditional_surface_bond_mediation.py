"""Test local bond bases on held-out readable rule-strength conflicts.

Fit the tag-contrast basis without any lexical-versus-induction conflicts.
Patch one attention-output bond from a Recall-tag donor into a Balanced-tag
recipient on identical body prefixes; report model-shift fidelity and a
no-effect frame-token control separately from oracle fidelity.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from conditional_surface_language import (MODES, ConditionalSurfaceLanguage)
from eval_conditional_bond_mediation import (forward_collect, forward_patch,
                                             make_bases, odds)


def data_with_positions(n, length, seed, regime='full'):
    gen = ConditionalSurfaceLanguage(
        seed=seed, mode_probs={mode: float(mode == 'Balanced') for mode in MODES},
        regime=regime)
    ids, labels, _ = gen.sample_batch(n, length+1)
    x = torch.tensor([row[:-1] for row in ids], dtype=torch.long)
    conflicts = torch.zeros((n, length), dtype=torch.bool)
    null = torch.zeros((n, length), dtype=torch.bool)
    positions = []
    for b, (row, labs) in enumerate(zip(ids, labels)):
        for t in range(1, length+1):
            if labs[t] == 'frame':
                null[b, t-1] = True
            if labs[t] != 'mixed_verb':
                continue
            candidates = gen.candidates(row[:t])
            if 'induction' not in candidates:
                continue
            lexical = max(candidates['lexical'], key=candidates['lexical'].get)
            induction = next(iter(candidates['induction']))
            if lexical == induction:
                continue
            conflicts[b, t-1] = True
            positions.append((b, t-1, gen.token2id[lexical],
                              gen.token2id[induction]))
    return x, positions, conflicts, null


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dir', required=True)
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--layer', type=int, default=0)
    p.add_argument('--nfit', type=int, default=256)
    p.add_argument('--ntest', type=int, default=512)
    p.add_argument('--length', type=int, default=71)
    p.add_argument('--regime', choices=('full', 'pair'), default='full')
    p.add_argument('--patch-positions', choices=('all', 'tag_only', 'non_tag_only'),
                   default='non_tag_only')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config

    torch.set_num_threads(4)
    saved = json.loads((Path(a.model_dir)/'results.json').read_text())
    assert saved['config'].get('regime', 'full') == a.regime, (
        'Model and evaluation regimes differ')
    record = next(row for row in saved['records'] if row['depth'] == a.depth)
    config = Config(**record['model_config'])
    model = BilinearGPT(config)
    model.load_state_dict(torch.load(Path(a.model_dir)/f'model_depth{a.depth}.pt',
                                     weights_only=True, map_location='cpu'))
    model.eval()
    bond = model.blocks[a.layer].attn.out
    fit_x, fit_positions, fit_conflict, _ = data_with_positions(
        a.nfit, a.length, 10601, a.regime)
    test_x, test_positions, _, test_null = data_with_positions(
        a.ntest, a.length, 10602, a.regime)
    donor_fit = fit_x.clone()
    donor_test = test_x.clone()
    mode_ids = ConditionalSurfaceLanguage().token2id
    donor_fit[:, 0] = mode_ids['Recall']
    donor_test[:, 0] = mode_ids['Recall']
    _, acts0 = forward_collect(model, bond, fit_x)
    _, acts1 = forward_collect(model, bond, donor_fit)
    fit_mask = ~fit_conflict
    fit_mask[:, 0] = False
    fit_delta = (acts1-acts0)[fit_mask].reshape(-1, config.n_embd).double()
    variance = torch.linalg.svdvals(fit_delta).square()
    if a.layer == 0:
        assert float(variance[2*config.n_head:].sum()/variance.sum()) < 1e-8
    spectra = {str(k): float(variance[:k].sum()/variance.sum())
               for k in (1, 2, 4, 8, 16, 32)}
    bases = make_bases(acts0, acts1, fit_mask, bond.weight.detach(),
                       config.n_head)
    logits0, _ = forward_collect(model, bond, test_x)
    logits1, donor_acts = forward_collect(model, bond, donor_test)
    reference = odds(logits1, test_positions)-odds(logits0, test_positions)
    assert len(reference) > 100
    logp0, logp1 = F.log_softmax(logits0, -1), F.log_softmax(logits1, -1)
    baseline_null_kl = float((logp0.exp()*(logp0-logp1)).sum(-1)[test_null].mean())
    rows = []
    for name, basis in bases.items():
        assert torch.linalg.matrix_norm(basis.T@basis-torch.eye(config.n_embd)) < 1e-4
        for width in (1, 2, 4, 8, 16, 32, config.n_embd):
            patched_logits = forward_patch(model, bond, test_x, donor_acts,
                                           basis, width, a.patch_positions)
            patched = odds(patched_logits, test_positions)-odds(
                logits0, test_positions)
            patch_logp = F.log_softmax(patched_logits, -1)
            null_kl = float((logp0.exp()*(logp0-patch_logp))
                            .sum(-1)[test_null].mean())
            rows.append(dict(basis=name, width=width,
                             n_conflicts=len(reference),
                             donor_shift_mean=float(reference.mean()),
                             patched_shift_mean=float(patched.mean()),
                             shift_rmse_to_donor=float(torch.sqrt(
                                 ((patched-reference)**2).mean())),
                             null_frame_patch_kl=null_kl))
    output = dict(config=vars(a), n_fit_conflicts=len(fit_positions),
                  n_test_conflicts=len(test_positions),
                  n_null_frame_positions=int(test_null.sum()),
                  no_patch_shift_rmse=float(reference.square().mean().sqrt()),
                  full_tag_null_frame_kl=baseline_null_kl,
                  contrast_variance_captured=spectra, rows=rows,
                  caveat='Contrast basis fitted only on no-conflict non-tag '
                         'positions. The patch uses the selected position '
                         'scope at one bond. '
                         'The donor model shift is a local mediation target, '
                         'not the exact DGP shift; see the separate fixed-prefix '
                         'oracle comparison. Full-width patch need not recover '
                         'the donor because residual and other paths remain.')
    Path(a.output).write_text(json.dumps(output, indent=2)+'\n')
    for row in rows:
        if row['width'] in (4, 16, config.n_embd) and row['basis'] in (
                'physical', 'activation_pca', 'weight_svd', 'tag_contrast_pca'):
            print(row['basis'], row['width'],
                  round(row['shift_rmse_to_donor'], 3), flush=True)


if __name__ == '__main__':
    main()

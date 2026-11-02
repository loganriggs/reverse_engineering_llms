"""Fit bond bases on one rule edit and test transfer to other edit types."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from eval_surface_mixture_edits import build_pairs
from surface_mixture_language import SurfaceMixtureLanguage


@torch.no_grad()
def infer(model, bond, prefixes, donor=None, basis=None, width=None,
          capture_full=False):
    probs = np.zeros((len(prefixes), model.config.vocab_size), dtype=np.float32)
    activations = ([None]*len(prefixes) if capture_full else
                   np.zeros((len(prefixes), model.config.n_embd), dtype=np.float32))
    groups = defaultdict(list)
    for i, prefix in enumerate(prefixes):
        groups[len(prefix)].append(i)
    for indices in groups.values():
        for start in range(0, len(indices), 32):
            batch_ids = indices[start:start+32]
            x = torch.tensor([prefixes[i] for i in batch_ids], dtype=torch.long)
            captured = []
            if donor is None:
                def capture(_, inputs):
                    captured.append(inputs[0].detach() if capture_full else
                                    inputs[0][:, -1].detach())
                hook = bond.register_forward_pre_hook(capture)
            else:
                donor_batch = torch.from_numpy(
                    np.stack([donor[i] for i in batch_ids]) if capture_full
                    else donor[batch_ids])
                uk = basis[:, :width]
                def patch(_, inputs):
                    z = inputs[0]
                    changed = z.clone()
                    if capture_full:
                        delta = donor_batch-z
                        changed = z+((delta@uk)@uk.T)
                    else:
                        delta = donor_batch-z[:, -1]
                        changed[:, -1] = z[:, -1]+((delta@uk)@uk.T)
                    return (changed,)
                hook = bond.register_forward_pre_hook(patch)
            logits = model(x)[0][:, -1].float()
            hook.remove()
            probs[batch_ids] = F.softmax(logits, -1).cpu().numpy()
            if donor is None:
                if capture_full:
                    for j, i in enumerate(batch_ids):
                        activations[i] = captured[0][j].cpu().numpy()
                else:
                    activations[batch_ids] = captured[0].cpu().numpy()
    return probs, activations


def make_bases(fit_pairs, fit_before, fit_after, weight, dimension):
    if isinstance(fit_before, list):
        before = np.concatenate(fit_before).astype(np.float64)
        delta = np.concatenate([after-before for before, after in
                                zip(fit_before, fit_after)]).astype(np.float64)
        labels = np.concatenate([[pair['kind']]*len(z)
                                 for pair, z in zip(fit_pairs, fit_before)])
    else:
        before = np.asarray(fit_before, dtype=np.float64)
        delta = np.asarray(fit_after-fit_before, dtype=np.float64)
        labels = np.asarray([pair['kind'] for pair in fit_pairs])
    def pca(z):
        _, u = np.linalg.eigh(z.T@z)
        return torch.tensor(np.flip(u, 1).copy(), dtype=torch.float32)
    _, _, vh = np.linalg.svd(weight.detach().cpu().double().numpy())
    random, _ = np.linalg.qr(np.random.default_rng(8319).normal(size=(dimension, dimension)))
    bases = dict(physical=torch.eye(dimension),
                 random=torch.tensor(random, dtype=torch.float32),
                 activation_pca=pca(before),
                 weight_svd=torch.tensor(vh.T.copy(), dtype=torch.float32))
    spectra = {}
    for kind in ('lead', 'induction', 'place_skip'):
        mask = labels == kind
        subset = delta[mask]
        bases[f'{kind}_contrast'] = pca(subset)
        singular = np.linalg.svd(subset, compute_uv=False)**2
        spectra[kind] = {str(k): float(singular[:k].sum()/singular.sum())
                         for k in (1, 2, 4, 8, 16, 32)}
    return bases, spectra


def score(test_pairs, base_p, donor_p, patched_p, full_p, vocab):
    results = {}
    for kind in ('lead', 'induction', 'place_skip', 'nuisance_adjective'):
        indices = np.asarray([i for i, pair in enumerate(test_pairs)
                              if pair['kind'] == kind])
        if not len(indices):
            continue
        model_effect = donor_p[indices]-base_p[indices]
        patch_effect = patched_p[indices]-base_p[indices]
        full_effect = full_p[indices]-base_p[indices]
        oracle_effect = np.zeros_like(model_effect)
        for j, index in enumerate(indices):
            pair = test_pairs[index]
            for token, mass in pair['q_after'].items():
                oracle_effect[j, vocab[token]] += mass
            for token, mass in pair['q_before'].items():
                oracle_effect[j, vocab[token]] -= mass
        results[kind] = dict(n=len(indices),
            model_effect_l1=float(np.abs(model_effect).sum(-1).mean()),
            full_bond_residual_l1=float(np.abs(full_effect-model_effect).sum(-1).mean()),
            patch_to_full_l1=float(np.abs(patch_effect-full_effect).sum(-1).mean()),
            patch_to_model_l1=float(np.abs(patch_effect-model_effect).sum(-1).mean()),
            patch_to_oracle_l1=float(np.abs(patch_effect-oracle_effect).sum(-1).mean()))
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dir', default='language_task_suite/surface_mixture_models_v1_rmsnorm')
    p.add_argument('--depth', type=int, default=4)
    p.add_argument('--layer', type=int, default=3)
    p.add_argument('--nfit', type=int, default=128)
    p.add_argument('--ntest', type=int, default=128)
    p.add_argument('--length', type=int, default=70)
    p.add_argument('--patch-scope', choices=('last', 'all'), default='last')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config

    torch.set_num_threads(4)
    saved = json.loads((Path(a.model_dir)/'results.json').read_text())
    record = next(row for row in saved['records'] if row['depth'] == a.depth)
    model = BilinearGPT(Config(**record['model_config']))
    model.load_state_dict(torch.load(Path(a.model_dir)/f'model_depth{a.depth}.pt',
                                     weights_only=True, map_location='cpu'))
    model.eval()
    bond = model.blocks[a.layer].attn.out
    fit_pairs = build_pairs(SurfaceMixtureLanguage(seed=8801), a.nfit, a.length)
    test_pairs = build_pairs(SurfaceMixtureLanguage(seed=8802), a.ntest, a.length)
    fit_base = [pair['original'] for pair in fit_pairs]
    fit_edit = [pair['edited'] for pair in fit_pairs]
    test_base = [pair['original'] for pair in test_pairs]
    test_edit = [pair['edited'] for pair in test_pairs]
    full = a.patch_scope == 'all'
    _, fit_before = infer(model, bond, fit_base, capture_full=full)
    _, fit_after = infer(model, bond, fit_edit, capture_full=full)
    bases, spectra = make_bases(fit_pairs, fit_before, fit_after,
                                bond.weight, model.config.n_embd)
    base_p, _ = infer(model, bond, test_base)
    donor_p, donor_acts = infer(model, bond, test_edit, capture_full=full)
    full_p, _ = infer(model, bond, test_base, donor_acts,
                      torch.eye(model.config.n_embd), model.config.n_embd,
                      capture_full=full)
    rows = []
    for name, basis in bases.items():
        assert torch.linalg.matrix_norm(basis.T@basis-torch.eye(basis.shape[1])) < 1e-4
        for width in (4, 8, 16, 32):
            patched_p, _ = infer(model, bond, test_base, donor_acts, basis, width,
                                 capture_full=full)
            rows.append(dict(basis=name, width=width,
                             groups=score(test_pairs, base_p, donor_p,
                                          patched_p, full_p,
                                          SurfaceMixtureLanguage().token2id)))
    output = dict(config=vars(a), n_fit_pairs=len(fit_pairs),
                  n_test_pairs=len(test_pairs),
                  contrast_spectra=spectra,
                  full_bond_groups=score(test_pairs, base_p, donor_p,
                                         full_p, full_p,
                                         SurfaceMixtureLanguage().token2id),
                  rows=rows,
                  caveat='Bond contrast bases fit on one edit family are tested '
                         'on separate sequences, including other edit families. '
                         'Full-width patch is a local mediation ceiling; residual '
                         'and other paths remain. No uniqueness claim.')
    Path(a.output).write_text(json.dumps(output, indent=2)+'\n')
    for row in rows:
        if row['width'] == 16:
            print(row['basis'], {kind: round(row['groups'][kind]['patch_to_full_l1'], 3)
                                  for kind in ('lead','induction','place_skip')})


if __name__ == '__main__':
    main()

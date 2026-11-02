"""Held-out alias intervention fidelity under final-bond truncation.

Fit bases on permitted alias-person pairings only. Test on the excluded pair
bucket, using first post-definition alias uses and exact supported edits.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from eval_surface_alias_edits import build_pairs
from surface_alias_mixture import AliasMixtureLanguage
from surface_mixture_v2 import VERBS


KINDS = ('same_referent', 'different_referent', 'definition_referent',
         'irrelevant_definition', 'definition_order', 'alias_label_swap')


def selected_pairs(gen, nstreams, length, max_per_kind, seed, holdout):
    all_pairs = build_pairs(gen, nstreams, length, 'balanced')
    rng = np.random.default_rng(seed)
    selected = []
    for kind in KINDS:
        available = [pair for pair in all_pairs
                     if pair['kind'] == kind
                     and pair['first_post_definition_use']
                     and pair['current_pair_holdout'] == holdout]
        if available:
            indices = rng.choice(len(available),
                                 min(max_per_kind, len(available)),
                                 replace=False)
            selected.extend(available[i] for i in indices)
    return selected


@torch.no_grad()
def capture(model, bond, seqs, device):
    result = np.zeros((len(seqs), model.config.n_embd), dtype=np.float32)
    groups = defaultdict(list)
    for i, seq in enumerate(seqs):
        groups[len(seq)].append(i)
    for _, indices in groups.items():
        for start in range(0, len(indices), 64):
            ix = indices[start:start+64]
            x = torch.tensor([seqs[i] for i in ix], dtype=torch.long,
                             device=device)
            captured = []
            hook = bond.register_forward_pre_hook(
                lambda _, inputs: captured.append(
                    inputs[0][:, -1].detach().float().cpu().numpy()))
            model(x)
            hook.remove()
            result[ix] = captured[0]
    return result


@torch.no_grad()
def predict(model, bond, seqs, device, directions=None, width=None):
    result = np.zeros((len(seqs), model.config.vocab_size), dtype=np.float32)
    groups = defaultdict(list)
    for i, seq in enumerate(seqs):
        groups[len(seq)].append(i)
    hook = None
    if directions is not None:
        uk = directions[:, :width].to(device)
        def truncate(_, inputs):
            x = inputs[0]
            out = x.clone()
            out[:, -1] = (x[:, -1]@uk)@uk.T
            return (out,)
        hook = bond.register_forward_pre_hook(truncate)
    for _, indices in groups.items():
        for start in range(0, len(indices), 64):
            ix = indices[start:start+64]
            x = torch.tensor([seqs[i] for i in ix], dtype=torch.long,
                             device=device)
            result[ix] = F.softmax(model(x)[0][:, -1].float(), -1).cpu().numpy()
    if hook is not None:
        hook.remove()
    return result


def bases(model, bond, fit, device):
    z0 = torch.from_numpy(capture(model, bond,
                                  [p['original'] for p in fit], device)).double()
    z1 = torch.from_numpy(capture(model, bond,
                                  [p['edited'] for p in fit], device)).double()
    centered = torch.cat([z0, z1])
    centered -= centered.mean(0, keepdim=True)
    _, _, data_vh = torch.linalg.svd(centered, full_matrices=False)
    _, _, response_vh = torch.linalg.svd(z1-z0, full_matrices=False)
    w = bond.weight.detach().float().cpu().double()
    head = model.lm_head.weight.detach().float().cpu().double()
    _, _, weight_vh = torch.linalg.svd(w, full_matrices=False)
    _, _, all_vh = torch.linalg.svd(head@w, full_matrices=False)
    # VERBS specifies the known task output set; this is a supervised prior.
    from surface_alias_mixture import TOKEN2ID
    _, _, verb_vh = torch.linalg.svd(head[[TOKEN2ID[v] for v in VERBS]]@w,
                                     full_matrices=True)
    random, _ = np.linalg.qr(np.random.default_rng(6511).normal(size=w.shape))
    return dict(weight_svd=weight_vh.T.float(),
                all_output=all_vh.T.float(),
                verb_output=verb_vh.T.float(),
                activation_pca=data_vh.T.float(),
                edit_response=response_vh.T.float(),
                random=torch.tensor(random, dtype=torch.float32))


def score(pairs, p0, p1, ref0, ref1, gen):
    result = {}
    for kind in KINDS:
        ix = [i for i, pair in enumerate(pairs) if pair['kind'] == kind]
        if not ix:
            result[kind] = dict(n=0)
            continue
        oracle = np.zeros_like(p0[ix])
        for j, i in enumerate(ix):
            for token, prob in pairs[i]['oracle_after'].items():
                oracle[j, gen.token2id[token]] += prob
            for token, prob in pairs[i]['oracle_before'].items():
                oracle[j, gen.token2id[token]] -= prob
        effect = p1[ix]-p0[ix]
        reference_effect = ref1[ix]-ref0[ix]
        result[kind] = dict(n=len(ix),
                            oracle_tv=float(np.abs(oracle).sum(-1).mean()/2),
                            model_tv=float(np.abs(effect).sum(-1).mean()/2),
                            oracle_effect_l1=float(np.abs(effect-oracle).sum(-1).mean()),
                            reference_effect_l1=float(np.abs(effect-reference_effect).sum(-1).mean()),
                            reference_tv=float((np.abs(p0[ix]-ref0[ix]).sum(-1).mean()
                                                +np.abs(p1[ix]-ref1[ix]).sum(-1).mean())/4))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--layer', type=int, default=1)
    p.add_argument('--fit-streams', type=int, default=1024)
    p.add_argument('--test-streams', type=int, default=1024)
    p.add_argument('--length', type=int, default=63)
    p.add_argument('--per-kind', type=int, default=192)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config

    torch.set_num_threads(4)
    saved = json.loads((a.model_dir/'results.json').read_text())
    if not saved['generator_kwargs'].get('exclude_alias_person_pairs'):
        raise ValueError('Model must have whole alias-person pair training holdout')
    record = next(row for row in saved['records'] if row['depth'] == 2)
    device = torch.device(a.device)
    model = BilinearGPT(Config(**record['model_config'])).to(device)
    model.load_state_dict(torch.load(a.model_dir/'model_depth2.pt',
                                     map_location=device, weights_only=True))
    model.eval()
    bond = model.blocks[a.layer].attn.out
    base = dict(initial_alias_definitions=2, parenthetical_rate=0,
                definition_start_rate=0, alias_use=.9, alias_reuse_rate=.8)
    fit_gen = AliasMixtureLanguage(seed=20261109,
                                   exclude_alias_person_pairs=True, **base)
    test_gen = AliasMixtureLanguage(seed=20261110, **base)
    fit = selected_pairs(fit_gen, a.fit_streams, a.length,
                         a.per_kind, 20261111, False)
    test = selected_pairs(test_gen, a.test_streams, a.length,
                          a.per_kind, 20261112, True)
    assert any(p['kind'] == 'different_referent' for p in fit)
    assert any(p['kind'] == 'different_referent' for p in test)
    basis = bases(model, bond, fit, device)
    before = [pair['original'] for pair in test]
    after = [pair['edited'] for pair in test]
    ref0 = predict(model, bond, before, device)
    ref1 = predict(model, bond, after, device)
    baseline = score(test, ref0, ref1, ref0, ref1, test_gen)
    rows = []
    for name, directions in basis.items():
        assert torch.linalg.matrix_norm(directions.T@directions-torch.eye(64)) < 1e-4
        for width in (0, 4, 8, 16, 32, 64):
            p0 = predict(model, bond, before, device, directions, width)
            p1 = predict(model, bond, after, device, directions, width)
            scores = score(test, p0, p1, ref0, ref1, test_gen)
            if width == 64:
                assert all(row['reference_tv'] < 1e-5
                           for row in scores.values() if row['n'])
            rows.append(dict(basis=name, width=width, scores=scores))
            key = scores['different_referent']
            print(name, width, round(key['oracle_effect_l1'], 3),
                  round(key['reference_tv'], 3), flush=True)
    result = dict(config={**vars(a), 'model_dir': str(a.model_dir),
                          'output': str(a.output)},
                  fit_cases=len(fit), test_cases=len(test),
                  baseline=baseline, rows=rows,
                  caveat='Local final-position bond truncation. Fit uses only '
                         'allowed-pair streams; test uses previously unseen '
                         'alias-person pairings. Full-width replay required. '
                         'Edit-response basis is explicitly supervised. '
                         'Compression is not selective circuit recovery.')
    a.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()

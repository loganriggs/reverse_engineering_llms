"""Compare local bond bases with Agreement held out from the basis fit.

The model was trained on all active genres. Only the activation-PCA basis fit
holds out an entire genre. Report both prediction and grammar-valid causal
edits; neither establishes discovery of a new task.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from eval_multi_genre_bond_gauges import group_scores
from eval_multi_genre_v3_agreement_edits import build_pairs
from eval_surface_bond_gauges import bases_for
from eval_surface_mixture_edits import predict, score
from multi_genre_v3_agreement import (GENRES_V3, MultiGenreV3Agreement)
from train_multi_genre import TASK_LABELS, make_batch, make_eval


AGREEMENT_LABELS = ('agreement_main_match', 'agreement_main_conflict',
                    'agreement_embedded')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dir', required=True)
    p.add_argument('--depth', type=int, default=3)
    p.add_argument('--layer', type=int, default=0)
    p.add_argument('--nfit', type=int, default=256)
    p.add_argument('--ntest', type=int, default=256)
    p.add_argument('--nedit', type=int, default=32)
    p.add_argument('--seed', type=int, default=9501,
                   help='Fit, prediction, and edit seeds use offsets 0/1/2')
    p.add_argument('--length', type=int, default=None)
    p.add_argument('--fit-genres', default='Reports,Guide,FAQ,Policy')
    p.add_argument('--allow-agreement-fit', action='store_true',
                   help='All-genre PCA exposure control, not a holdout')
    p.add_argument('--widths', default='8,16,32,48,64')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config

    torch.set_num_threads(4)
    saved = json.loads((Path(a.model_dir)/'results.json').read_text())
    assert saved['config']['language'] == 'v3'
    record = next(r for r in saved['records'] if r['depth'] == a.depth)
    model = BilinearGPT(Config(**record['model_config']))
    model.load_state_dict(torch.load(
        Path(a.model_dir)/f'model_depth{a.depth}.pt',
        weights_only=True, map_location='cpu'))
    model.eval()
    bond = model.blocks[a.layer].attn.out
    length = a.length or saved['config']['length']
    fit_genres = set(a.fit_genres.split(','))
    assert fit_genres and fit_genres.issubset(GENRES_V3)
    assert 'Agreement' not in fit_genres or a.allow_agreement_fit
    fit_weights = tuple(float(g in fit_genres) for g in GENRES_V3)
    xfit, _ = make_batch(MultiGenreV3Agreement(seed=a.seed,
                                               genre_weights=fit_weights),
                         a.nfit, length, 'cpu')
    train_weights = tuple(float(w) for w in
                          saved['config']['genre_weights'].split(','))
    assert train_weights[-1] > 0
    test = make_eval(MultiGenreV3Agreement(seed=a.seed+1,
                                            genre_weights=train_weights),
                     a.ntest, length, 'cpu',
                     TASK_LABELS + AGREEMENT_LABELS, GENRES_V3)
    pairs = build_pairs(MultiGenreV3Agreement(
        seed=a.seed+2, genre_weights=(0, 0, 0, 0, 0, 1)), a.nedit, length)
    assert pairs
    originals = [pair['original'] for pair in pairs]
    edits = [pair['edited'] for pair in pairs]
    captured = []
    hook = bond.register_forward_pre_hook(
        lambda _, inputs: captured.append(inputs[0].detach()))
    with torch.no_grad():
        for batch in xfit.split(32):
            model(batch)
    hook.remove()
    fit_activations = torch.cat(captured)
    captured.clear()
    agreement_x, _ = make_batch(MultiGenreV3Agreement(
        seed=a.seed+3, genre_weights=(0, 0, 0, 0, 0, 1)),
        a.nfit, length, 'cpu')
    hook = bond.register_forward_pre_hook(
        lambda _, inputs: captured.append(inputs[0].detach()))
    with torch.no_grad():
        for batch in agreement_x.split(32):
            model(batch)
    hook.remove()
    agreement_activations = torch.cat(captured)
    bases = bases_for(fit_activations, bond.weight.detach())
    with torch.no_grad():
        reference_logits = torch.cat([model(batch)[0].float()
                                      for batch in test['x'].split(32)])
    reference_logp = F.log_softmax(reference_logits, -1)
    native_before = predict(model, originals, len(saved['vocabulary']))
    native_after = predict(model, edits, len(saved['vocabulary']))
    native_edits = score(pairs, native_before, native_after,
                         MultiGenreV3Agreement(seed=1))
    widths = sorted({int(w) for w in a.widths.split(',')})
    assert widths[-1] == model.config.n_embd
    rows = []
    for name, basis in bases.items():
        assert torch.linalg.matrix_norm(basis.T@basis-torch.eye(
            basis.shape[1])) < 1e-4
        for width in widths:
            u = basis[:, :width]
            def captured_energy(z):
                z = z.reshape(-1, z.shape[-1]).double()
                u64 = u.double()
                return float((z@u64).square().sum()/z.square().sum())
            fit_energy = captured_energy(fit_activations)
            agreement_energy = captured_energy(agreement_activations)
            weight_energy = captured_energy(bond.weight.detach())
            def truncate(_, inputs):
                x = inputs[0]
                return ((x@u)@u.T,)
            hook = bond.register_forward_pre_hook(truncate)
            with torch.no_grad():
                logits = torch.cat([model(batch)[0].float()
                                    for batch in test['x'].split(32)])
            before = predict(model, originals, len(saved['vocabulary']))
            after = predict(model, edits, len(saved['vocabulary']))
            hook.remove()
            if width == model.config.n_embd:
                assert torch.max(torch.abs(logits-reference_logits)) < 3e-4
                assert torch.max(torch.abs(torch.from_numpy(before-native_before))) < 1e-4
            rows.append(dict(basis=name, width=width,
                             fit_activation_energy=fit_energy,
                             agreement_activation_energy=agreement_energy,
                             output_weight_energy=weight_energy,
                             groups=group_scores(logits, test, reference_logp),
                             edits=score(pairs, before, after,
                                         MultiGenreV3Agreement(seed=1))))
            print(name, width, 'agreement model KL',
                  round(rows[-1]['groups']['agreement_main_conflict']
                        ['model_to_truncated_kl'], 5),
                  'subject-edit L1',
                  round(rows[-1]['edits']['main_subject_flip']
                        ['effect_l1_error'], 5), flush=True)
    result = dict(config=vars(a), model_config=record['model_config'],
                  n_pairs=len(pairs), native_edits=native_edits, rows=rows,
                  caveat=('Agreement was exposed to the PCA basis fit. '
                          if 'Agreement' in fit_genres else
                          'Agreement was held out from the PCA basis fit, '
                          'not from model training. ') +
                         'Truncation tests local '
                         'compression, not a unique computation or new-task '
                         'discovery.')
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()

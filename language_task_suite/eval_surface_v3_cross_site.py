"""Test whether local rule-response directions transfer from verbs to objects."""

import argparse
import json
import sys
from pathlib import Path

import torch

from eval_surface_mixture_edits import predict
from eval_surface_mixture_v2_edits import build_pairs as build_verb_pairs
from eval_surface_mixture_v3_edits import build_object_pairs
from eval_surface_v2_rule_mediation import mean_l1, mean_tv, predict_patched
from eval_surface_v2_rule_subspaces import (FIT_FRAMES, TEST_FRAMES,
                                            energy, normalized_svd_basis,
                                            response_deltas)
from surface_mixture_v3 import SurfaceMixtureV3


KINDS = ('induction', 'object_induction', 'place_skip', 'object_event_skip')


def make_pairs(seed, n, length, frames):
    verb = build_verb_pairs(SurfaceMixtureV3(seed=seed), n, length)
    obj = build_object_pairs(SurfaceMixtureV3(seed=seed), n, length)
    return [pair for pair in verb+obj
            if pair['kind'] in KINDS and pair['frame'] in frames]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dir', default='language_task_suite/surface_mixture_v3_soft_depth3')
    p.add_argument('--depth', type=int, default=3)
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

    folder = Path(a.model_dir)
    saved = json.loads((folder/'results.json').read_text())
    record = next(row for row in saved['records'] if row['depth'] == a.depth)
    model = BilinearGPT(Config(**record['model_config']))
    model.load_state_dict(torch.load(folder/f'model_depth{a.depth}.pt',
                                     map_location='cpu', weights_only=True))
    model.eval()
    fit_frames = TEST_FRAMES if a.reverse_frames else FIT_FRAMES
    test_frames = FIT_FRAMES if a.reverse_frames else TEST_FRAMES
    fit = make_pairs(9901, a.nfit, a.length, fit_frames)
    test = make_pairs(8701, a.ntest, a.length, test_frames)
    original = [pair['original'] for pair in test]
    edited = [pair['edited'] for pair in test]
    p0 = torch.from_numpy(predict(model, original, model.config.vocab_size))
    p1 = torch.from_numpy(predict(model, edited, model.config.vocab_size))
    results = {}
    for layer in range(a.depth):
        bond = model.blocks[layer].attn.out
        fit_delta = response_deltas(model, bond, fit)
        test_delta = response_deltas(model, bond, test)
        fit_by_kind = {kind: fit_delta[[i for i, pair in enumerate(fit)
                                       if pair['kind'] == kind]] for kind in KINDS}
        bases = {kind: normalized_svd_basis(rows, a.width)
                 for kind, rows in fit_by_kind.items()}
        _, _, vh = torch.linalg.svd(bond.weight.detach().double(),
                                   full_matrices=False)
        bases['weight_svd'] = vh[:a.width].T.float()
        full = predict_patched(model, bond, original, test_delta)
        layer_results = {}
        for kind in KINDS:
            indices = [i for i, pair in enumerate(test) if pair['kind'] == kind]
            d = test_delta[indices]
            prefixes = [original[i] for i in indices]
            full_effect = full[indices]-p0[indices]
            model_effect = p1[indices]-p0[indices]
            row = dict(n=len(indices),
                       model_effect_tv=mean_tv(p1[indices], p0[indices]),
                       full_patch_tv=mean_tv(full[indices], p0[indices]),
                       full_patch_to_model_l1=mean_l1(full_effect, model_effect),
                       bases={})
            for basis_name, basis in bases.items():
                projected = (d@basis)@basis.T
                patched = predict_patched(model, bond, prefixes, projected)
                row['bases'][basis_name] = dict(
                    response_energy=energy(d, basis)['mean'],
                    patch_tv=mean_tv(patched, p0[indices]),
                    patch_to_full_l1=mean_l1(patched-p0[indices], full_effect))
            layer_results[kind] = row
        results[str(layer)] = layer_results
        print('layer', layer,
              {k: (round(v['model_effect_tv'], 3),
                   round(v['full_patch_tv'], 3))
               for k, v in layer_results.items()}, flush=True)
    output = dict(config=vars(a), fit_frames=fit_frames,
                  test_frames=test_frames,
                  fit_counts={k: sum(p['kind'] == k for p in fit) for k in KINDS},
                  test_counts={k: sum(p['kind'] == k for p in test) for k in KINDS},
                  results=results,
                  caveat='A verb-trained local response basis is tested on an '
                         'object task held out from featurizer fitting, not from '
                         'model training. Low cross-site transfer at one bond '
                         'does not rule out a shared algorithm elsewhere. Full '
                         'local patch mediation must be checked before scores '
                         'for a projected basis are interpreted.')
    Path(a.output).write_text(json.dumps(output, indent=2)+'\n')


if __name__ == '__main__':
    main()

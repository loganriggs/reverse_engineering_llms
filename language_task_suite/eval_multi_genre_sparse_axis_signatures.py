"""Post-fit task signatures of sparse gauge axes on disjoint edit streams."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from eval_multi_genre_edits import build_pairs
from eval_multi_genre_sparse_gauges import fit_bases
from eval_surface_edit_bases import infer
from multi_genre_language import MultiGenreLanguage


CAUSAL = ('policy_condition', 'faq_answer_copy', 'faq_place_recall')
NUISANCE = 'policy_nuisance_person'


def score_axes(pairs, delta, basis):
    labels = np.asarray([pair['kind'] for pair in pairs])
    coordinates = delta@basis
    energy = {kind: np.mean(coordinates[labels == kind]**2, axis=0)
              for kind in CAUSAL+(NUISANCE,)
              if np.any(labels == kind)}
    normalized = np.stack([energy[kind]/max(energy[kind].sum(), 1e-30)
                           for kind in CAUSAL])
    purity = float(normalized.max(axis=0).sum()/normalized.sum())
    families = {}
    for kind, e in energy.items():
        descending = np.sort(e)[::-1]
        families[kind] = dict(total_energy=float(e.sum()),
                              top1_fraction=float(descending[:1].sum()/e.sum()),
                              top4_fraction=float(descending[:4].sum()/e.sum()),
                              top8_fraction=float(descending[:8].sum()/e.sum()),
                              largest_axes=np.argsort(e)[-8:][::-1].tolist())
    return dict(equal_causal_family_axis_purity=purity, families=families)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--nfit', type=int, default=512)
    parser.add_argument('--nedit-streams', type=int, default=512)
    parser.add_argument('--length', type=int, default=71)
    parser.add_argument('--fit-genres', default='Reports,Policy')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, args.source)
    from model import BilinearGPT, Config
    torch.set_num_threads(4)
    saved = json.loads((Path(args.model_dir)/'results.json').read_text())
    record = next(row for row in saved['records'] if row['depth'] == args.depth)
    model = BilinearGPT(Config(**record['model_config']))
    model.load_state_dict(torch.load(Path(args.model_dir)/f'model_depth{args.depth}.pt',
                                     weights_only=True, map_location='cpu'))
    model.eval()
    bond = model.blocks[args.layer].attn.out
    bases, fit = fit_bases(model, bond, args.nfit, args.length, 9551,
                           set(args.fit_genres.split(',')))
    gen = MultiGenreLanguage(seed=9553)
    pairs = [pair for pair in build_pairs(gen, args.nedit_streams, args.length)
             if pair['kind'] in CAUSAL+(NUISANCE,)]
    assert all(any(pair['kind'] == kind for pair in pairs)
               for kind in CAUSAL+(NUISANCE,))
    _, before = infer(model, bond, [p['original'] for p in pairs])
    _, after = infer(model, bond, [p['edited'] for p in pairs])
    delta = (after-before).astype(np.float64)
    rows = {name: score_axes(pairs, delta, basis.double().numpy())
            for name, basis in bases.items()}
    result = dict(config={**vars(args), 'output': str(args.output)}, fit=fit,
                  pair_counts=dict(Counter(pair['kind'] for pair in pairs)),
                  rows=rows,
                  caveat='Labels and edits are post-fit diagnostics only. '
                         'The basis is fitted on specified genres; FAQ is a '
                         'whole-genre holdout when absent from fit. Energy '
                         'at a bond can reflect noncausal input differences; '
                         'consult full-bond mediation and oracle-effect tests '
                         'before assigning component meaning.')
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    for name, row in rows.items():
        print(name, round(row['equal_causal_family_axis_purity'], 3),
              {kind: round(row['families'][kind]['top8_fraction'], 3)
               for kind in CAUSAL}, flush=True)


if __name__ == '__main__':
    main()

"""Final-bond basis fidelity for source edits on withheld speaker-object pairs.

Fit data-aware bases on the restricted pair DGP. Evaluate on an independent,
unrestricted stream, selecting only questions whose queried statement pair was
withheld during model training. A local pre-output bond projection is not a
claim about a unique or globally canonical source circuit.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from eval_alias_bond_bases import capture, predict
from eval_surface_mixture_v6_sources import build
from surface_mixture_v6 import SPEAKERS, SurfaceMixtureV6


KINDS = ('query_object', 'statement_speaker', 'other_speaker')


def basis_set(model, bond, fit_pairs, stream_prefixes, device, seed):
    original = [pair['original'] for pair in fit_pairs]
    edited = [pair['edited'] for pair in fit_pairs]
    z0 = torch.from_numpy(capture(model, bond, original, device)).double()
    z1 = torch.from_numpy(capture(model, bond, edited, device)).double()
    centered = torch.cat([z0, z1])
    centered -= centered.mean(0, keepdim=True)
    _, _, pca_vh = torch.linalg.svd(centered, full_matrices=True)
    _, _, response_vh = torch.linalg.svd(z1-z0, full_matrices=True)
    stream_z = torch.from_numpy(capture(model, bond, stream_prefixes,
                                        device)).double()
    stream_z -= stream_z.mean(0, keepdim=True)
    _, _, stream_vh = torch.linalg.svd(stream_z, full_matrices=True)
    w = bond.weight.detach().float().cpu().double()
    head = model.lm_head.weight.detach().float().cpu().double()
    _, _, weight_vh = torch.linalg.svd(w, full_matrices=True)
    _, _, all_vh = torch.linalg.svd(head@w, full_matrices=True)
    gen = SurfaceMixtureV6()
    speaker_rows = [gen.token2id[s] for s in SPEAKERS]
    _, _, speaker_vh = torch.linalg.svd(head[speaker_rows]@w,
                                        full_matrices=True)
    random, _ = np.linalg.qr(np.random.default_rng(seed).normal(
        size=(w.shape[1], w.shape[1])))
    return dict(weight_svd=weight_vh.T.float(),
                all_output=all_vh.T.float(),
                speaker_output=speaker_vh.T.float(),
                stream_pca=stream_vh.T.float(),
                activation_pca=pca_vh.T.float(),
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
            for token, mass in pairs[i]['q_after'].items():
                oracle[j, gen.token2id[token]] += mass
            for token, mass in pairs[i]['q_before'].items():
                oracle[j, gen.token2id[token]] -= mass
        effect = p1[ix]-p0[ix]
        reference = ref1[ix]-ref0[ix]
        result[kind] = dict(
            n=len(ix),
            oracle_tv=float(np.abs(oracle).sum(-1).mean()/2),
            model_tv=float(np.abs(effect).sum(-1).mean()/2),
            oracle_effect_l1=float(np.abs(effect-oracle).sum(-1).mean()),
            reference_effect_l1=float(np.abs(effect-reference).sum(-1).mean()),
            reference_tv=float((np.abs(p0[ix]-ref0[ix]).sum(-1).mean()+
                                np.abs(p1[ix]-ref1[ix]).sum(-1).mean())/4))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--config-json', type=Path, required=True,
                   help='Training results.json with model_config or records')
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--layer', type=int, default=1)
    p.add_argument('--fit-streams', type=int, default=1000)
    p.add_argument('--test-streams', type=int, default=2000)
    p.add_argument('--length', type=int, default=128)
    p.add_argument('--fit-seed', type=int, default=24101)
    p.add_argument('--test-seed', type=int, default=24102)
    p.add_argument('--random-seed', type=int, default=24103)
    p.add_argument('--stream-pca-seed', type=int, default=24107)
    p.add_argument('--stream-pca-streams', type=int, default=512)
    p.add_argument('--stream-pca-per-stream', type=int, default=8)
    p.add_argument('--device', default='cpu')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config

    torch.set_num_threads(4)
    saved = json.loads(a.config_json.read_text())
    config = (saved['model_config'] if 'model_config' in saved else
              next(record['model_config'] for record in saved['records']
                   if record['depth'] == 2))
    assert config['n_layer'] == 2
    device = torch.device(a.device)
    model = BilinearGPT(Config(**config)).to(device)
    model.load_state_dict(torch.load(a.checkpoint, map_location=device,
                                     weights_only=True))
    model.eval()
    bond = model.blocks[a.layer].attn.out
    common = dict(quote_rate=.30, source_rate=.30, question_rate=.05,
                  counterbalanced_queries=True)
    fit_gen = SurfaceMixtureV6(seed=a.fit_seed,
                               exclude_source_pairs=True, **common)
    test_gen = SurfaceMixtureV6(seed=a.test_seed, **common)
    _, fit_pairs, _ = build(fit_gen, a.fit_streams, a.length)
    _, all_test_pairs, _ = build(test_gen, a.test_streams, a.length)
    test_pairs = [pair for pair in all_test_pairs if pair['pair_holdout']]
    assert fit_pairs and test_pairs
    assert all(not pair['pair_holdout'] for pair in fit_pairs)
    assert {pair['kind'] for pair in test_pairs} == set(KINDS)

    stream_gen = SurfaceMixtureV6(seed=a.stream_pca_seed,
                                  exclude_source_pairs=True, **common)
    stream_ids, _, _ = stream_gen.sample_batch(a.stream_pca_streams,
                                              a.length+1)
    stream_rng = np.random.default_rng(a.stream_pca_seed+1)
    positions = stream_rng.integers(16, a.length+1,
                                    size=(a.stream_pca_streams,
                                          a.stream_pca_per_stream))
    stream_prefixes = [row[:int(position)] for row, chosen in
                       zip(stream_ids, positions) for position in chosen]
    directions = basis_set(model, bond, fit_pairs, stream_prefixes,
                           device, a.random_seed)
    before = [pair['original'] for pair in test_pairs]
    after = [pair['edited'] for pair in test_pairs]
    ref0 = predict(model, bond, before, device)
    ref1 = predict(model, bond, after, device)
    baseline = score(test_pairs, ref0, ref1, ref0, ref1, test_gen)
    rows = []
    for name, matrix in directions.items():
        assert torch.linalg.matrix_norm(matrix.T@matrix-
                                        torch.eye(config['n_embd'])) < 1e-4
        for width in (0, 4, 8, 16, 32, 64):
            p0 = predict(model, bond, before, device, matrix, width)
            p1 = predict(model, bond, after, device, matrix, width)
            scores = score(test_pairs, p0, p1, ref0, ref1, test_gen)
            if width == config['n_embd']:
                assert all(value['reference_tv'] < 1e-5
                           for value in scores.values() if value['n'])
            rows.append(dict(basis=name, width=width, scores=scores))
            key = scores['query_object']
            print(name, width, round(key['oracle_effect_l1'], 4),
                  round(key['reference_tv'], 4), flush=True)
    result = dict(config={**vars(a),
                          'config_json': str(a.config_json),
                          'checkpoint': str(a.checkpoint),
                          'output': str(a.output)},
                  fit_pairs=len(fit_pairs),
                  stream_pca_prefixes=len(stream_prefixes),
                  test_pairs=len(test_pairs),
                  test_kind_counts={kind: sum(pair['kind'] == kind
                                              for pair in test_pairs)
                                    for kind in KINDS},
                  baseline=baseline, rows=rows,
                  caveat='Model task seen in training. Data bases fit only on '
                         'allowed speaker-object pairings; test queries '
                         'withheld pairings. Speaker-output and edit-response '
                         'bases use task knowledge; activation PCA is '
                         'source-edit-selected, stream PCA samples unlabeled '
                         'positions. Local final-position bond '
                         'projection measures fidelity, not circuit recovery.')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()

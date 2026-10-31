"""Exact unseen-query test for weight-only versus dataset-weighted reductions.

The model has already learned both query forms. Only the reduction fit omits
the swapped query, so this tests coverage of a post-training representation.
"""

import json
from itertools import combinations

import numpy as np
import torch

from analyze_dual_output_language import factors, score
from analyze_toy_language import pair_table
from dual_joint_function_canonical import (HERE, flatten, matched_cosines,
                                           rotations, unflatten)
from dual_joint_metric import weighted_fit
from train_dual_output_language import DualOutputAttention, exhaustive


RARE_MASSES = (0.0001, 0.001, 0.01, 0.05, 0.5)
ANGLES, _ = rotations()
PAIRS = np.asarray(tuple(combinations(range(6), 2)))


def weight(rare_mass):
    w = np.empty((2, 2, 6))
    w[0] = (1 - rare_mass) / 12
    w[1] = rare_mass / 12
    return w.reshape(4, 6)


def exact_common_completions(matrix):
    """Two rank-two cuts identical on all seen cells, different on unseen rows."""
    common = matrix[:2]
    u, s, v = np.linalg.svd(common, full_matrices=False)
    dictionary = np.sqrt(s[:, None]) * v
    left = u * np.sqrt(s)
    assert np.max(np.abs(left @ dictionary - common)) < 1e-10
    zero = np.vstack((common, np.zeros_like(common)))
    # Use the original *weight table* to choose rare-query coefficients in the
    # common-row span. This is a weight-informed completion, not a data-only fit.
    rare_left = matrix[2:] @ dictionary.T @ np.linalg.inv(dictionary @ dictionary.T)
    informed = np.vstack((common, rare_left @ dictionary))
    assert np.linalg.matrix_rank(zero, tol=1e-8) <= 2
    assert np.linalg.matrix_rank(informed, tol=1e-8) <= 2
    assert np.max(np.abs(zero[:2] - informed[:2])) < 1e-12
    return zero, informed


def gauge_scores(cut):
    """Select exact rotations without task labels; score known variables later."""
    u, s, v = np.linalg.svd(cut, full_matrices=False)
    left = u[:, :2] * np.sqrt(s[:2])
    dictionary = np.sqrt(s[:2, None]) * v[:2]
    candidates = np.einsum('aij,jk->aik', ANGLES, dictionary)
    key_l1 = np.abs(candidates).sum(axis=(1, 2))
    differences = candidates[:, :, PAIRS[:, 1]] - candidates[:, :, PAIRS[:, 0]]
    edit_l1 = np.abs(differences).sum(axis=1).mean(axis=1)
    selected = {'svd': 0, 'key_l1': int(np.argmin(key_l1)),
                'all_pair_edit_l1': int(np.argmin(edit_l1))}
    result = {}
    for name, index in selected.items():
        d = candidates[index]
        l = left @ ANGLES[index].T
        assert np.max(np.abs(l @ d - cut)) < 1e-9
        result[name] = dict(variable_alignment=matched_cosines(d)[0],
                            key_l1=float(key_l1[index]),
                            edit_l1=float(edit_l1[index]))
    return result


def run():
    torch.set_num_threads(1)
    seq_t, targets_t, attrs = exhaustive()
    seq = seq_t.numpy()
    targets = targets_t.numpy()
    rows = []
    for seed in range(10):
        model = DualOutputAttention().double()
        model.load_state_dict(torch.load(HERE / f'dual_output_seed{seed}.pt',
                                         map_location='cpu', weights_only=True))
        model.eval()
        heads = [factors(model, h) for h in range(2)]
        original = flatten(sum(pair_table(h, np.eye(h['rank'])) for h in heads))
        bias = model.read.bias.detach().numpy()
        native = score([unflatten(original)], seq, targets, bias, attrs)[0]
        assert native['joint_accuracy'] == 1
        assert native['rare_heldout']['joint_accuracy'] == 1
        zero, informed = exact_common_completions(original)
        cuts = {'seen_only_zero_completion': zero,
                'seen_only_weight_informed_completion': informed}
        for rare_mass in RARE_MASSES:
            cut, fit = weighted_fit(original, weight(rare_mass), 2)
            assert fit['optimality'] < 1e-5
            cuts[f'weighted_rare_mass_{rare_mass:g}'] = cut
        for name, cut in cuts.items():
            result = score([unflatten(cut)], seq, targets, bias, attrs)[0]
            rows.append(dict(seed=seed, reduction=name,
                observed_common_table_mse=float(np.mean((cut[:2]-original[:2])**2)),
                unseen_rare_table_mse=float(np.mean((cut[2:]-original[2:])**2)),
                broad_table_relative_mse=float(np.sum((cut-original)**2)/np.sum(original**2)),
                common_joint_accuracy=result['common']['joint_accuracy'],
                rare_seen_joint_accuracy=result['rare_seen']['joint_accuracy'],
                rare_heldout_joint_accuracy=result['rare_heldout']['joint_accuracy'],
                full_joint_accuracy=result['joint_accuracy'],
                gauges=gauge_scores(cut)))
        print('seed', seed, flush=True)
    output = dict(description='Ten trained dual-output bilinear-attention models: zero-support common-query-only data fit has multiple equal-error rank-two completions; compare zero and weight-informed completions with positive rare-query mass and exact-gauge legibility priors.',
        rare_masses=RARE_MASSES, model_training_uses_both_query_forms=True,
        reduction_fit_uses_answer_labels=False,
        query_exclusion_applies_only_to_reduction_fit=True,
        weight_informed_completion_uses_unseen_weight_rows=True, rows=rows)
    path = HERE / 'dual_unseen_query_coverage.json'
    path.write_text(json.dumps(output, indent=2) + '\n')
    for name in sorted({r['reduction'] for r in rows}):
        part = [r for r in rows if r['reduction'] == name]
        print(name, 'common', np.mean([r['common_joint_accuracy'] for r in part]),
              'rare holdout', np.mean([r['rare_heldout_joint_accuracy'] for r in part]),
              'rare table MSE', np.mean([r['unseen_rare_table_mse'] for r in part]),
              'edit alignment', np.mean([r['gauges']['all_pair_edit_l1']['variable_alignment']
                                         for r in part]), flush=True)


if __name__ == '__main__':
    run()

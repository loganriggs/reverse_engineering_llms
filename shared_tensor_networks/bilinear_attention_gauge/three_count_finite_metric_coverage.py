"""Finite unlabeled sequence coverage for rank cuts of trained bilinear toys."""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from analyze_three_count_language import flatten, head_tables
from dual_joint_function_canonical import HERE
from three_count_metric import weighted_fit
from three_count_rank_selection import task_scores
from train_three_count_composed import ComposedCountModel
from train_three_count_language import PAD, examples


ALL_PAIRS = np.asarray(tuple(combinations(range(8), 2)))
WITHIN_PAIRS = np.asarray(((0, 1), (2, 3), (4, 5), (6, 7)))


def edit_error(reference, cut, pairs):
    ref = reference.reshape(2, 3, 8)
    approx = cut.reshape(2, 3, 8)
    target = ref[:, :, pairs[:, 1]] - ref[:, :, pairs[:, 0]]
    predicted = approx[:, :, pairs[:, 1]] - approx[:, :, pairs[:, 0]]
    return [float(np.sum((predicted[q]-target[q])**2)/
                  np.sum(target[q]**2)) for q in (0, 1)]


def empirical_weights(tokens):
    query = tokens[:, -1] - 8
    keys = tokens[:, :-1]
    valid = keys != PAD
    counts = np.zeros((2, 8), dtype=np.float64)
    np.add.at(counts, (np.broadcast_to(query[:, None], keys.shape)[valid],
                       keys[valid]), 1)
    repeated = np.repeat(counts, 3, axis=0)
    return repeated / repeated.sum(), counts


def rank_svd(matrix, rank):
    u, s, v = np.linalg.svd(matrix, full_matrices=False)
    return (u[:, :rank] * s[:rank]) @ v[:rank]


def one_model(seed, sizes, repeats):
    model = ComposedCountModel().double()
    model.load_state_dict(torch.load(
        HERE / f'three_count_composed_scratch_seed{seed}.pt',
        map_location='cpu', weights_only=True))
    model.eval()
    matrix = flatten(sum(head_tables(model.backbone)))
    scale = model.logit_scale.detach().numpy()
    quadratic = model.quadratic.detach().numpy()
    quadratic = .5 * (quadratic + quadratic.T) / scale[:, None] / scale[None, :]
    bias = model.backbone.read.bias.detach().numpy()
    parity_bias = float(model.parity_bias.detach())
    test_t, targets_t, attrs = examples(30000, 130000 + seed, False)
    test, targets = test_t.numpy(), targets_t.numpy()
    reference = task_scores(matrix, quadratic, bias, parity_bias,
                            test, targets, attrs)
    assert reference['rare_heldout']['all_four_accuracy'] == 1
    uniform = np.ones_like(matrix) / matrix.size
    records = []
    for rank in (2, 3):
        cut = rank_svd(matrix, rank)
        metrics = task_scores(cut, quadratic, bias, parity_bias,
                              test, targets, attrs)
        records.append(dict(seed=seed, sample_size=None, repeat=None,
            rank=rank, measure='uniform_weight_only',
            rare_query_count=None, covered_query_key_cells=16,
            covered_rare_key_cells=8, fit_nfev=0,
            full_relative_table_mse=float(np.sum((cut-matrix)**2)/np.sum(matrix**2)),
            all_pair_edit_error_by_query=edit_error(matrix, cut, ALL_PAIRS),
            within_role_edit_error_by_query=edit_error(matrix, cut, WITHIN_PAIRS),
            task=metrics))
    for n in sizes:
        for repeat in range(repeats):
            sampled, _, _ = examples(n, 1000000+seed*10000+repeat*100+n, True)
            sampled = sampled.numpy()
            empirical, counts = empirical_weights(sampled)
            rare_count = int(np.sum(sampled[:, -1] == 9))
            for alpha, name in ((0., 'empirical'),
                                (.05, 'floor_0.05'),
                                (.5, 'floor_0.5')):
                fit_weight = (1-alpha)*empirical + alpha*uniform
                for rank in (2, 3):
                    cut, fit = weighted_fit(matrix, fit_weight, rank)
                    metrics = task_scores(cut, quadratic, bias, parity_bias,
                                          test, targets, attrs)
                    records.append(dict(seed=seed, sample_size=n, repeat=repeat,
                        rank=rank, measure=name,
                        rare_query_count=rare_count,
                        covered_query_key_cells=int(np.sum(counts>0)),
                        covered_rare_key_cells=int(np.sum(counts[1]>0)),
                        fit_nfev=fit['nfev'],
                        fit_optimality=fit['optimality'],
                        full_relative_table_mse=float(
                            np.sum((cut-matrix)**2)/np.sum(matrix**2)),
                        all_pair_edit_error_by_query=edit_error(matrix, cut, ALL_PAIRS),
                        within_role_edit_error_by_query=edit_error(matrix, cut, WITHIN_PAIRS),
                        task=metrics))
    return records


def run(args):
    torch.set_num_threads(1)
    sizes = tuple(int(x) for x in args.sizes.split(','))
    records = []
    for seed in range(args.seeds):
        records.extend(one_model(seed, sizes, args.repeats))
        print('seed', seed, 'complete', flush=True)
    result = dict(description='Finite unlabeled usage measures and uniform floors for rank-two/three exact token-function cuts of ten trained three-bit plus parity bilinear-attention models',
        model_seeds=args.seeds, data_repeats=args.repeats,
        sequence_sample_sizes=sizes, test_sequences_per_model=30000,
        fitting_uses_full_weight_derived_token_table=True,
        sampled_sequences_determine_only_fit_weights=True,
        weighted_solver_starts_from_full_table_svd=True,
        model_training_uses_task_labels=True,
        reduction_fit_uses_task_labels=False,
        rank_selection_edit_panels_use_task_labels=False,
        records=records)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    for n in sizes:
        for rank in (2, 3):
            for measure in ('empirical', 'floor_0.05', 'floor_0.5'):
                part = [r for r in records if r['sample_size']==n and
                        r['rank']==rank and r['measure']==measure]
                print(n, rank, measure, 'rare', round(np.mean([
                    r['task']['rare_heldout']['all_four_accuracy'] for r in part]), 4),
                    'table', round(np.mean([r['full_relative_table_mse'] for r in part]), 4),
                    'coverage', round(np.mean([r['covered_rare_key_cells'] for r in part]), 2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--sizes', default='8,32,128,512,2048,8192')
    parser.add_argument('--output', type=Path,
                        default=HERE/'three_count_finite_metric_coverage.json')
    run(parser.parse_args())

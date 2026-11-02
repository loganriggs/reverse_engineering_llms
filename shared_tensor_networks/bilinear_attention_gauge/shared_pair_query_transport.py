"""Recover query transport of primitive bit logits from trained weights.

The rare query rotates the target bit order B/C/A. This analysis asks whether
the same permutation emerges without labels from the exact learned attention
tables, and whether it predicts parity coefficients and held-out responses.
"""

import itertools
import json

import numpy as np
import torch

from analyze_three_count_language import flatten, head_tables
from dual_joint_function_canonical import HERE
from shared_pair_data_equivalent_node_edits import sample
from shared_pair_physical_node_alignment import semantic_forms
from train_shared_pair_count import SharedPairCountModel
from train_shared_pair_dense import examples_dense
from train_three_count_language import examples as examples_paired


CODINGS = (('paired', 'shared_pair_count', 'shared_pair_gauge', examples_paired),
           ('dense', 'shared_pair_dense', 'shared_pair_dense_gauge',
            examples_dense))
PERMUTATIONS = tuple(itertools.permutations(range(3)))
DESIGNED_PERMUTATION = (1, 2, 0)


def fit_permutation(common, rare, permutation):
    permuted = common[list(permutation)]
    scales = np.sum(rare*permuted, axis=1) / np.sum(permuted**2, axis=1)
    predicted = scales[:, None]*permuted
    relative_mse = float(np.sum((predicted-rare)**2) /
                         np.sum(rare**2))
    return relative_mse, scales, predicted


def parity_coefficient_error(common, rare, transported, q):
    residual, target = 0., 0.
    for p in range(2):
        true = rare.T @ q[p] @ rare
        predicted = transported.T @ q[p] @ transported
        residual += float(np.sum((predicted-true)**2))
        target += float(np.sum(true**2))
    return residual/max(target, 1e-20)


def run():
    torch.set_num_threads(1)
    rows = []
    for coding, prefix, gauge_name, source in CODINGS:
        analysis = json.loads((HERE/f'{prefix}_analysis.json').read_text())
        gauge = json.loads((HERE/f'{gauge_name}.json').read_text())
        for seed in range(10):
            model = SharedPairCountModel().double()
            model.load_state_dict(torch.load(HERE/f'{prefix}_seed{seed}.pt',
                                  map_location='cpu', weights_only=True))
            h = np.stack([flatten(t) for t in
                          head_tables(model.backbone)]).sum(axis=0)
            common, rare = h.reshape(2, 3, 8)
            fits = [(perm, *fit_permutation(common, rare, perm))
                    for perm in PERMUTATIONS]
            fits.sort(key=lambda item: item[1])
            best_perm, best_error, scales, predicted = fits[0]
            second_error = fits[1][1]
            assert np.all(scales > 0)
            record = next(r for r in analysis['rows'] if r['seed']==seed)
            q = np.asarray(record['quadratic_effective'])
            coefficient_error = parity_coefficient_error(common, rare,
                                                           predicted, q)
            counts, query, attrs = sample(source, 30000, 1010000+seed,
                                          False)
            rare_query = query == 1
            rare_holdout = attrs['heldout']
            true = counts @ rare.T
            transported = counts @ predicted.T

            def signal_error(mask):
                return float(np.sum((transported[mask]-true[mask])**2) /
                             max(float(np.sum(true[mask]**2)), 1e-20))

            gauge_row = next(r for r in gauge['rows'] if r['seed']==seed
                             and r['reduction']=='uniform_rank3')
            semantic, labels, _ = semantic_forms(
                np.asarray(record['matrices']['uniform_rank3']), q,
                np.asarray(gauge_row['gauges']['joint_nullspace']['dictionary']))
            cell_summaries = {}
            for index, label in enumerate(labels):
                if tuple(label) not in ((0,1), (0,2), (1,2)):
                    continue
                energy = np.sum(semantic[index]**2, axis=(2,3))
                cell_summaries[str(tuple(label))] = dict(
                    energy_fraction=(energy/np.sum(energy)).tolist(),
                    dominant_cell=[int(x) for x in np.unravel_index(
                        np.argmax(energy), energy.shape)])
            rows.append(dict(coding=coding, seed=seed,
                best_permutation=list(best_perm),
                best_weight_relative_mse=best_error,
                second_best_weight_relative_mse=second_error,
                row_scales=scales.tolist(),
                parity_coefficient_relative_mse=coefficient_error,
                rare_query_signal_relative_mse=signal_error(rare_query),
                rare_heldout_signal_relative_mse=signal_error(rare_holdout),
                n_rare_heldout=int(rare_holdout.sum()),
                semantic_node_cell_summaries=cell_summaries))
            print(coding, seed, flush=True)
    summary = {}
    for coding, _, _, _ in CODINGS:
        part = [r for r in rows if r['coding']==coding]
        assert len(part)==10
        summary[coding] = dict(
            designed_permutation_recovered=sum(tuple(r['best_permutation'])==
                DESIGNED_PERMUTATION for r in part),
            mean_best_weight_relative_mse=float(np.mean([
                r['best_weight_relative_mse'] for r in part])),
            mean_second_best_weight_relative_mse=float(np.mean([
                r['second_best_weight_relative_mse'] for r in part])),
            mean_row_scales=np.mean([r['row_scales'] for r in part],
                                    axis=0).tolist(),
            mean_parity_coefficient_relative_mse=float(np.mean([
                r['parity_coefficient_relative_mse'] for r in part])),
            mean_rare_heldout_signal_relative_mse=float(np.mean([
                r['rare_heldout_signal_relative_mse'] for r in part])))
    result = dict(description='Discover the common-to-rare query permutation and row gains by exhaustive six-permutation weight-only fit to exact trained head tables. Test the transported bit response on independent rare held-out task combinations, and its induced quadratic parity coefficients on all real count vectors. No task labels enter permutation selection.',
        designed_permutation=list(DESIGNED_PERMUTATION), rows=rows,
        summary=summary)
    (HERE/'shared_pair_query_transport.json').write_text(
        json.dumps(result, indent=2)+'\n')
    for coding, stats in summary.items():
        print(coding, stats, flush=True)
    return result


if __name__=='__main__':
    run()

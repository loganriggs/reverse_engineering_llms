"""Can physical head-pair effects imitate semantic node deletions on task data?

The physical comparator is deliberately permissive: any *global linear
combination* of six exact head-pair parity-response terms, fitted by least
squares. It need not correspond to a feasible head or branch patch.
"""

import json

import numpy as np
import torch
from scipy.optimize import minimize

from analyze_three_count_language import flatten, head_tables
from dual_joint_function_canonical import HERE
from shared_pair_physical_node_alignment import physical_forms, semantic_forms
from three_count_task_fit import count_features
from train_shared_pair_count import SharedPairCountModel
from train_shared_pair_dense import examples_dense
from train_three_count_language import examples as examples_paired


CODINGS = (('paired', 'shared_pair_count', examples_paired),
           ('dense', 'shared_pair_dense', examples_dense))
GAUGES = ('svd', 'joint_nullspace')
N_FIT = 20000
N_TEST = 30000


def evaluate_forms(forms, counts, query):
    """Return [sentence, node, parity_output] quadratic contributions."""
    out = np.empty((len(counts), len(forms), 2), dtype=np.float64)
    for query_id in (0, 1):
        mask = query == query_id
        out[mask] = np.einsum('ni,hpij,nj->nhp', counts[mask],
                              forms[:, :, query_id], counts[mask])
    return out


def sample(source, n, seed, exclude_rare):
    tokens, _, attrs = source(n, seed, exclude_rare)
    counts = count_features(tokens.numpy()).astype(np.float64)
    query = (tokens[:, -1]-8).numpy().astype(int)
    assert set(np.unique(query)) == {0, 1}
    if exclude_rare:
        assert not attrs['heldout'].any()
    return counts, query, attrs


def fit_physical_span(physical, target):
    # Important: transpose before flattening, so each sentence/output row
    # contains the six physical predictors for that SAME sentence/output.
    design = physical.transpose(0, 2, 1).reshape(-1, 6)
    response = target.reshape(-1)
    scale = np.sqrt(np.mean(design**2, axis=0))
    assert np.all(scale > 1e-12)
    normalized = design / scale
    coefficient = np.linalg.lstsq(normalized, response,
                                  rcond=None)[0] / scale
    return coefficient


def predict_physical(physical, coefficient):
    return np.einsum('nhp,h->np', physical, coefficient)


def fit_head_attenuation(physical, target, labels):
    """Fit an executable edit: multiply each of three head outputs by g in [0,1]."""
    design = physical.transpose(0, 2, 1).reshape(-1, 6)
    response = target.reshape(-1)
    gram = (design.T @ design) / len(response)
    cross = (design.T @ response) / len(response)
    assert len(labels) == 6

    def coefficients(gains):
        return np.asarray([1-gains[i]*gains[j] for i, j in labels])

    def objective(gains):
        alpha = coefficients(gains)
        return float(alpha @ gram @ alpha-2*alpha @ cross)

    starts = [np.ones(3), np.zeros(3), np.full(3, .5)]
    starts += [np.eye(3)[i] for i in range(3)]
    fits = [minimize(objective, start, bounds=[(0, 1)]*3,
                     method='L-BFGS-B', options={'ftol': 1e-13})
            for start in starts]
    best = min(fits, key=lambda result: result.fun)
    return coefficients(best.x), best.x


def metrics(target, prediction, rare, baseline_parity):
    residual = prediction-target
    denom = max(float(np.mean(target**2)), 1e-20)
    seen = ~rare
    semantic_deleted = baseline_parity-target
    physical_deleted = baseline_parity-prediction
    flips = np.any((semantic_deleted > 0) != (baseline_parity > 0), axis=1)
    return dict(full_relative_mse=float(np.mean(residual**2)/denom),
        seen_relative_mse=float(np.mean(residual[seen]**2) /
                                max(float(np.mean(target[seen]**2)), 1e-20)),
        rare_relative_mse=float(np.mean(residual[rare]**2) /
                                max(float(np.mean(target[rare]**2)), 1e-20)),
        rare_mse_over_full_target_energy=float(
            np.mean(residual[rare]**2)/denom),
        rare_target_energy_over_full=float(
            np.mean(target[rare]**2)/denom),
        deleted_parity_sign_agreement=float(np.mean(
            np.all((semantic_deleted > 0) == (physical_deleted > 0), axis=1))),
        semantic_deletion_flip_count=int(flips.sum()),
        sign_agreement_on_semantic_flips=(
            float(np.mean(np.all(
                (semantic_deleted[flips] > 0) ==
                (physical_deleted[flips] > 0), axis=1)))
            if flips.any() else None))


def baseline_parity(matrix, q, bias, parity_bias, counts, query):
    table = matrix.reshape(2, 3, 8)
    signal = np.einsum('nk,nok->no', counts, table[query])
    bits = signal+bias
    return np.einsum('ni,pij,nj->np', bits, q, bits)+parity_bias


def run():
    torch.set_num_threads(1)
    rows = []
    for coding, prefix, source in CODINGS:
        analysis = json.loads((HERE / f'{prefix}_analysis.json').read_text())
        gauge_file = ('shared_pair_gauge.json' if coding == 'paired'
                      else 'shared_pair_dense_gauge.json')
        gauge_data = json.loads((HERE / gauge_file).read_text())
        for seed in range(10):
            model = SharedPairCountModel().double()
            model.load_state_dict(torch.load(
                HERE / f'{prefix}_seed{seed}.pt', map_location='cpu',
                weights_only=True))
            model.eval()
            heads = np.stack([flatten(t).reshape(2, 3, 8)
                              for t in head_tables(model.backbone)])
            analysis_row = next(x for x in analysis['rows'] if
                                x['seed'] == seed)
            q = np.asarray(analysis_row['quadratic_effective'])
            physical_forms_array, physical_labels = physical_forms(heads, q)
            matrix = np.asarray(
                analysis_row['matrices']['uniform_rank3'])
            gauge_row = next(x for x in gauge_data['rows'] if
                             x['seed'] == seed and
                             x['reduction'] == 'uniform_rank3')
            fit_counts, fit_query, _ = sample(
                source, N_FIT, 710000+seed, True)
            oracle_counts, oracle_query, _ = sample(
                source, N_FIT, 720000+seed, False)
            test_counts, test_query, attrs = sample(
                source, N_TEST, 730000+seed, False)
            rare = attrs['heldout']
            assert int(rare.sum()) > 100
            physical_fit = evaluate_forms(
                physical_forms_array, fit_counts, fit_query)
            physical_oracle = evaluate_forms(
                physical_forms_array, oracle_counts, oracle_query)
            physical_test = evaluate_forms(
                physical_forms_array, test_counts, test_query)
            bias = model.backbone.read.bias.detach().numpy()
            parity_bias = model.parity_bias.detach().numpy()
            base = baseline_parity(matrix, q, bias, parity_bias,
                                   test_counts, test_query)
            for gauge in GAUGES:
                dictionary = np.asarray(
                    gauge_row['gauges'][gauge]['dictionary'])
                semantic, labels, coefficients = semantic_forms(
                    matrix, q, dictionary)
                semantic_fit = evaluate_forms(
                    semantic, fit_counts, fit_query)
                semantic_oracle = evaluate_forms(
                    semantic, oracle_counts, oracle_query)
                semantic_test = evaluate_forms(
                    semantic, test_counts, test_query)
                top = np.argsort(
                    np.sum(coefficients**2, axis=(0, 1)))[-3:]
                for index in top:
                    target_fit = semantic_fit[:, index]
                    target_oracle = semantic_oracle[:, index]
                    target_test = semantic_test[:, index]
                    seen_coefficient = fit_physical_span(
                        physical_fit, target_fit)
                    oracle_coefficient = fit_physical_span(
                        physical_oracle, target_oracle)
                    physical_train_2d = physical_fit.transpose(
                        0, 2, 1).reshape(-1, 6)
                    target_train_1d = target_fit.reshape(-1)
                    one_coefficients = (
                        np.sum(physical_train_2d *
                               target_train_1d[:, None], axis=0) /
                        np.sum(physical_train_2d**2, axis=0))
                    one_train_mse = np.mean(
                        (physical_train_2d *
                         one_coefficients[None, :] -
                         target_train_1d[:, None])**2, axis=0)
                    best_one = int(np.argmin(one_train_mse))
                    single_coefficient = np.zeros(6)
                    single_coefficient[best_one] = one_coefficients[best_one]
                    head_coefficient, head_gains = fit_head_attenuation(
                        physical_fit, target_fit, physical_labels)
                    variants = {
                        'best_single_seen': single_coefficient,
                        'three_head_attenuation_seen': head_coefficient,
                        'six_pair_span_seen': seen_coefficient,
                        'six_pair_span_oracle_full': oracle_coefficient}
                    scores = {name: metrics(target_test,
                        predict_physical(physical_test, coef), rare, base)
                        for name, coef in variants.items()}
                    rows.append(dict(coding=coding, seed=seed, gauge=gauge,
                        semantic_node=list(labels[index]),
                        best_single_physical_pair=list(
                            physical_labels[best_one]),
                        n_fit=N_FIT, n_test=N_TEST,
                        n_rare=int(rare.sum()),
                        fitted_head_gains=head_gains.tolist(),
                        coefficients={name:coef.tolist()
                                      for name,coef in variants.items()},
                        scores=scores))
            print(coding, seed, 'done', flush=True)
    summary = {}
    for coding, _, _ in CODINGS:
        for gauge in GAUGES:
            part = [r for r in rows if r['coding'] == coding
                    and r['gauge'] == gauge]
            assert len(part) == 30
            summary[f'{coding}__{gauge}'] = {
                name: {field: float(np.mean([
                    row['scores'][name][field] for row in part]))
                    for field in ('full_relative_mse',
                                  'seen_relative_mse',
                                  'rare_mse_over_full_target_energy',
                                  'deleted_parity_sign_agreement')}
                for name in ('best_single_seen', 'six_pair_span_seen',
                             'six_pair_span_oracle_full',
                             'three_head_attenuation_seen')}
    result = dict(description='Best scalar physical head-pair term, feasible attenuation of three physical head outputs, unrestricted linear combination of all six physical head-pair terms fitted on seen task data, and full-distribution oracle fit. Predict semantic-node deletion effects on independent task sentences and rare held-out combination. Fits use no answer labels; full-distribution fit is a permissive positive control, not a valid holdout method.',
        n_fit=N_FIT, n_test=N_TEST, codings=[c for c,_,_ in CODINGS],
        gauges=GAUGES, rows=rows, summary=summary)
    path = HERE / 'shared_pair_data_equivalent_node_edits.json'
    path.write_text(json.dumps(result, indent=2)+'\n')
    for name, values in summary.items():
        print(name, {method:{key:round(val,3) for key,val in metrics.items()}
                     for method,metrics in values.items()}, flush=True)
    return result


if __name__ == '__main__':
    run()

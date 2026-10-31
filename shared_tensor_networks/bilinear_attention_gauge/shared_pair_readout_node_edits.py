"""Can physical quadratic-readout edits realize semantic product deletions?

The attention backbone produces three primitive logits b. Both parity outputs
depend only on b via symmetric quadratic matrices Q. Editing Q leaves b
exactly unchanged, unlike any upstream patch that changes parity.
"""

import json

import numpy as np
import torch

from analyze_three_count_language import flatten, head_tables
from dual_joint_function_canonical import HERE
from shared_pair_data_equivalent_node_edits import sample, evaluate_forms
from shared_pair_physical_node_alignment import semantic_forms
from train_shared_pair_count import SharedPairCountModel
from train_shared_pair_dense import examples_dense
from train_three_count_language import examples as examples_paired


N_FIT = 20000
N_TEST = 30000
CODINGS = (('paired', 'shared_pair_count', examples_paired),
           ('dense', 'shared_pair_dense', examples_dense))
PAIRS = tuple((i, j) for i in range(3) for j in range(i, 3))


def readout_basis():
    result = []
    for i, j in PAIRS:
        q = np.zeros((3, 3))
        q[i, j] = 1.
        q[j, i] = 1.
        result.append(q)
    return np.stack(result)


Q_BASIS = readout_basis()


def bit_logits(matrix, bias, counts, query):
    tables = matrix.reshape(2, 3, 8)
    return np.einsum('nk,nbk->nb', counts, tables[query]) + bias


def design_from_bits(bits):
    return np.einsum('ni,kij,nj->nk', bits, Q_BASIS, bits)


def weight_only_design(matrix, bias):
    """Polynomial coefficients: constant, linear, quadratic for each query."""
    h = matrix.reshape(2, 3, 8)
    cols = []
    for basis in Q_BASIS:
        query_terms = []
        for query in range(2):
            constant = bias @ basis @ bias
            linear = 2*bias @ basis @ h[query]
            quadratic = h[query].T @ basis @ h[query]
            query_terms.append(np.concatenate(([constant], linear,
                                                quadratic.ravel())))
        cols.append(np.concatenate(query_terms))
    return np.stack(cols, axis=1)


def weight_only_design_by_query(matrix, bias):
    """Polynomial coefficient design for a separate edit at each query."""
    h = matrix.reshape(2, 3, 8)
    blocks = []
    for query in range(2):
        cols = []
        for basis in Q_BASIS:
            constant = bias @ basis @ bias
            linear = 2*bias @ basis @ h[query]
            quadratic = h[query].T @ basis @ h[query]
            cols.append(np.concatenate(([constant], linear,
                                         quadratic.ravel())))
        blocks.append(np.stack(cols, axis=1))
    return np.stack(blocks)


def weight_only_target(form_for_parity):
    return np.concatenate([np.concatenate((np.zeros(9), form.ravel()))
                           for form in form_for_parity])


def fit_target(design, target):
    coefficients, _, rank, singular = np.linalg.lstsq(design, target,
                                                       rcond=None)
    return coefficients, int(rank), singular


def metric(target, pred, rare):
    err = (pred-target)**2
    energy = max(float(np.mean(target**2)), 1e-20)
    return dict(full_relative_mse=float(np.mean(err)/energy),
        seen_relative_mse=float(np.mean(err[~rare]) /
            max(float(np.mean(target[~rare]**2)), 1e-20)),
        rare_mse_over_full_target_energy=float(np.mean(err[rare])/energy),
        rare_relative_mse=float(np.mean(err[rare]) /
            max(float(np.mean(target[rare]**2)), 1e-20)))


def run():
    torch.set_num_threads(1)
    rows = []
    for coding, prefix, source in CODINGS:
        analysis = json.loads((HERE/f'{prefix}_analysis.json').read_text())
        gauge_data = json.loads((HERE/('shared_pair_gauge.json' if coding ==
                    'paired' else 'shared_pair_dense_gauge.json')).read_text())
        for seed in range(10):
            model = SharedPairCountModel().double()
            model.load_state_dict(torch.load(HERE/f'{prefix}_seed{seed}.pt',
                                  map_location='cpu', weights_only=True))
            model.eval()
            record = next(r for r in analysis['rows'] if r['seed'] == seed)
            q_effective = np.asarray(record['quadratic_effective'])
            reduced = np.asarray(record['matrices']['uniform_rank3'])
            gauge_row = next(r for r in gauge_data['rows'] if r['seed'] == seed
                             and r['reduction'] == 'uniform_rank3')
            dictionary = np.asarray(
                gauge_row['gauges']['joint_nullspace']['dictionary'])
            semantic, labels, coefficients = semantic_forms(
                reduced, q_effective, dictionary)
            selected = np.argsort(np.sum(coefficients**2, axis=(0, 1)))[-3:]
            matrix = np.stack([flatten(t) for t in
                               head_tables(model.backbone)]).sum(axis=0)
            bias = model.backbone.read.bias.detach().numpy()
            uniform_design = weight_only_design(matrix, bias)
            uniform_query_design = weight_only_design_by_query(matrix, bias)
            fit_counts, fit_query, _ = sample(source, N_FIT,
                                               910000+seed, True)
            test_counts, test_query, attrs = sample(source, N_TEST,
                                                    920000+seed, False)
            assert int(attrs['heldout'].sum()) > 100
            fit_bits = bit_logits(matrix, bias, fit_counts, fit_query)
            test_bits = bit_logits(matrix, bias, test_counts, test_query)
            fit_design, test_design = (design_from_bits(fit_bits),
                                       design_from_bits(test_bits))
            fit_targets = evaluate_forms(semantic, fit_counts, fit_query)
            test_targets = evaluate_forms(semantic, test_counts, test_query)
            # Check exact replay of the native quadratic readout on direct
            # model outputs, including scaling and the primitive read bias.
            tokens, _, _ = source(128, 930000+seed, False)
            counts = np.zeros((128, 8), dtype=np.float64)
            for key in range(8):
                counts[:, key] = (tokens.numpy()[:, :-1] == key).sum(1)
            query = (tokens[:, -1]-8).numpy()
            native_bits = bit_logits(matrix, bias, counts, query)
            with torch.no_grad():
                direct = model(tokens).numpy()
            assert np.max(np.abs(native_bits-direct[:, :3])) < 1e-9
            parity = np.einsum('ni,pij,nj->np', native_bits,
                               q_effective, native_bits)
            parity += model.parity_bias.detach().numpy()
            assert np.max(np.abs(parity-direct[:, 3:])) < 1e-7
            for index in selected:
                target_fit, target_test = fit_targets[:, index], test_targets[:, index]
                data_coeff, rank, singular = [], [], []
                uniform_coeff = []
                conditioned_data_coeff = []
                conditioned_uniform_coeff = []
                for p in range(2):
                    data, data_rank, data_singular = fit_target(
                        fit_design, target_fit[:, p])
                    uniform, uniform_rank, uniform_singular = fit_target(
                        uniform_design, weight_only_target(semantic[index,p]))
                    assert data_rank == 6 and uniform_rank == 6
                    data_coeff.append(data)
                    uniform_coeff.append(uniform)
                    query_data, query_uniform = [], []
                    for query_id in (0, 1):
                        mask = fit_query == query_id
                        conditioned_data, conditional_rank, _ = fit_target(
                            fit_design[mask], target_fit[mask, p])
                        conditioned_uniform, _, _ = fit_target(
                            uniform_query_design[query_id],
                            np.concatenate((np.zeros(9),
                                semantic[index, p, query_id].ravel())))
                        assert conditional_rank == 6
                        query_data.append(conditioned_data)
                        query_uniform.append(conditioned_uniform)
                    conditioned_data_coeff.append(np.stack(query_data))
                    conditioned_uniform_coeff.append(np.stack(query_uniform))
                    rank.append(data_rank)
                    singular.append(data_singular.tolist())
                data_coeff, uniform_coeff = (np.stack(data_coeff),
                                             np.stack(uniform_coeff))
                if index == selected[0]:
                    # Verify that the fitted effective-Q edit is an actual
                    # parameter patch of the trained quadratic readout.
                    delta_effective = np.einsum('pk,kij->pij',
                                                 data_coeff, Q_BASIS)
                    scale = model.logit_scale.detach().numpy()
                    delta_raw = (delta_effective * scale[None, :, None] *
                                 scale[None, None, :])
                    original_quadratic = model.quadratic.detach().clone()
                    with torch.no_grad():
                        model.quadratic.add_(torch.from_numpy(delta_raw))
                        patched = model(tokens).numpy()
                        model.quadratic.copy_(original_quadratic)
                    expected = design_from_bits(native_bits) @ data_coeff.T
                    assert np.max(np.abs(patched[:, :3]-direct[:, :3])) < 1e-9
                    assert np.max(np.abs(
                        (patched-direct)[:, 3:]-expected)) < 1e-7
                conditioned_data_coeff = np.stack(conditioned_data_coeff)
                conditioned_uniform_coeff = np.stack(conditioned_uniform_coeff)
                conditional_data = conditioned_data_coeff[:, test_query].transpose(1, 0, 2)
                conditional_uniform = conditioned_uniform_coeff[:, test_query].transpose(1, 0, 2)
                predictions = {
                    'seen_data_readout_fit': test_design @ data_coeff.T,
                    'weight_only_readout_fit': test_design @ uniform_coeff.T,
                    'query_conditioned_seen_fit': np.einsum(
                        'nk,npk->np', test_design, conditional_data),
                    'query_conditioned_weight_only_fit': np.einsum(
                        'nk,npk->np', test_design, conditional_uniform)}
                coefficient_errors = {}
                for name, fitted in (
                    ('weight_only_readout_fit', uniform_coeff),
                    ('query_conditioned_weight_only_fit',
                     conditioned_uniform_coeff)):
                    squared_residual, squared_target = 0., 0.
                    for p in range(2):
                        true = weight_only_target(semantic[index, p])
                        if name == 'weight_only_readout_fit':
                            predicted = uniform_design @ fitted[p]
                        else:
                            predicted = np.concatenate([
                                uniform_query_design[query_id] @
                                fitted[p, query_id] for query_id in (0, 1)])
                        squared_residual += float(np.sum((predicted-true)**2))
                        squared_target += float(np.sum(true**2))
                    coefficient_errors[name] = squared_residual/max(
                        squared_target, 1e-20)
                scores = {name: metric(target_test, pred,
                    attrs['heldout']) for name, pred in predictions.items()}
                rows.append(dict(coding=coding, seed=seed,
                    semantic_node=list(labels[index]), n_fit=N_FIT,
                    n_test=N_TEST, n_rare=int(attrs['heldout'].sum()),
                    coefficients={
                        'seen_data_readout_fit': data_coeff.tolist(),
                        'weight_only_readout_fit': uniform_coeff.tolist(),
                        'query_conditioned_seen_fit': conditioned_data_coeff.tolist(),
                        'query_conditioned_weight_only_fit': conditioned_uniform_coeff.tolist()},
                    readout_basis_pairs=[list(x) for x in PAIRS],
                    fit_design_rank=rank, fit_singular_values=singular,
                    exact_coefficient_relative_mse=coefficient_errors,
                    scores=scores))
            print(coding, seed, flush=True)
    summary = {}
    for coding, _, _ in CODINGS:
        for node in ((0, 1), (0, 2), (1, 2)):
            subset = [r for r in rows if r['coding'] == coding and
                      tuple(r['semantic_node']) == node]
            assert len(subset) == 10
            summary[f'{coding}__{node}'] = {fit: {
                field: float(np.mean([r['scores'][fit][field]
                                      for r in subset]))
                for field in ('full_relative_mse',
                              'rare_mse_over_full_target_energy')}
                for fit in ('seen_data_readout_fit',
                            'weight_only_readout_fit',
                            'query_conditioned_seen_fit',
                            'query_conditioned_weight_only_fit')}
            summary[f'{coding}__{node}']['coefficient_errors'] = {
                fit: float(np.mean([r['exact_coefficient_relative_mse'][fit]
                                    for r in subset]))
                for fit in ('weight_only_readout_fit',
                            'query_conditioned_weight_only_fit')}
    result = dict(description='Physical symmetric quadratic-readout edits as candidates for semantic product-node deletions. Compare exact weight-only coefficient fit with seen-task-data fit and query-conditioned diagnostics; independent full and rare test. Only the query-independent edit exists in the trained architecture. Primitive bit logits are invariant by construction. No task answers used.',
        n_fit=N_FIT, n_test=N_TEST, rows=rows, summary=summary)
    (HERE/'shared_pair_readout_node_edits.json').write_text(
        json.dumps(result, indent=2)+'\n')
    for name, value in summary.items():
        print(name, {fit:{k:round(v,4) for k,v in metrics.items()}
                     for fit,metrics in value.items()}, flush=True)
    return result


if __name__ == '__main__':
    run()

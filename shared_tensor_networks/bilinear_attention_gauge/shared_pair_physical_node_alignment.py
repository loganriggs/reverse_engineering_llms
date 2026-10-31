"""Compare recovered quadratic function nodes with physical head-pair terms.

Each trained three-head model has a quadratic parity readout. Expanding it
over the three physical attention-head contributions gives six exact
head-pair forms. A sparse gauge of a rank-three ordinary-function cut gives
six semantic coordinate monomials; earlier work keeps its three strongest.
Here we compare these forms in the full query/output/token-count coefficient
space, without using task sentences or fitting a matching matrix.
"""

import json

import numpy as np
import torch

from analyze_three_count_language import flatten, head_tables
from dual_joint_function_canonical import HERE
from shared_pair_gauge import monomial_strengths
from train_shared_pair_count import SharedPairCountModel


GAUGES = ('svd', 'key_l1', 'joint_nullspace', 'cell_one_only')
CODINGS = (('paired', 'shared_pair_count'), ('dense', 'shared_pair_dense'))


def physical_forms(head_matrix, q):
    """Six [parity,query,key,key] symmetric quadratic coefficient tensors."""
    assert head_matrix.shape == (3, 2, 3, 8)
    forms = []
    labels = []
    for i in range(3):
        for j in range(i, 3):
            block = np.empty((2, 2, 8, 8))
            for p in range(2):
                for query in range(2):
                    raw = head_matrix[i, query].T @ q[p] @ head_matrix[j, query]
                    block[p, query] = ((raw + raw.T) / 2 if i == j
                                       else raw + raw.T)
            forms.append(block)
            labels.append((i, j))
    return np.stack(forms), labels


def semantic_forms(matrix, q, dictionary):
    coefficients, labels = monomial_strengths(matrix, q, dictionary)
    forms = []
    for node, (i, j) in enumerate(labels):
        outer = np.outer(dictionary[i], dictionary[j])
        basis = outer if i == j else (outer + outer.T) / np.sqrt(2)
        forms.append(coefficients[:, :, node, None, None] * basis)
    return np.stack(forms), labels, coefficients


def total_form(matrix, q):
    response = matrix.reshape(2, 3, 8)
    return np.einsum('qia,pij,qjb->pqab', response, q, response)


def run():
    torch.set_num_threads(1)
    rows = []
    for coding, prefix in CODINGS:
        analysis = json.loads((HERE / f'{prefix}_analysis.json').read_text())
        gauge_data = json.loads((HERE / (
            'shared_pair_gauge.json' if coding == 'paired' else
            'shared_pair_dense_gauge.json')).read_text())
        for seed in range(10):
            model = SharedPairCountModel().double()
            model.load_state_dict(torch.load(
                HERE / f'{prefix}_seed{seed}.pt', map_location='cpu',
                weights_only=True))
            model.eval()
            heads = np.stack([flatten(t).reshape(2, 3, 8)
                              for t in head_tables(model.backbone)])
            record = next(x for x in analysis['rows'] if x['seed'] == seed)
            q = np.asarray(record['quadratic_effective'])
            physical, physical_labels = physical_forms(heads, q)
            original = total_form(heads.sum(axis=0).reshape(6, 8), q)
            native_gate = float(np.max(np.abs(physical.sum(axis=0)-original)))
            assert native_gate < 1e-7, (coding, seed, native_gate)
            physical_flat = physical.reshape(6, -1)
            _, singular, right = np.linalg.svd(physical_flat,
                                               full_matrices=False)
            assert singular[-1] > 1e-8 * singular[0]
            physical_span = right.T @ right
            physical_norms = np.linalg.norm(physical_flat, axis=1)
            assert np.all(physical_norms > 0)
            physical_cells = np.sum(physical**2, axis=(-1, -2))
            physical_cell_purity = np.max(physical_cells.reshape(6, -1), axis=1) / (
                np.sum(physical_cells.reshape(6, -1), axis=1))

            reduced = np.asarray(record['matrices']['uniform_rank3'])
            gauge_row = next(x for x in gauge_data['rows'] if
                             x['seed'] == seed and
                             x['reduction'] == 'uniform_rank3')
            for name in GAUGES:
                dictionary = np.asarray(
                    gauge_row['gauges'][name]['dictionary'])
                semantic, semantic_labels, coefficients = semantic_forms(
                    reduced, q, dictionary)
                reduced_gate = float(np.max(np.abs(
                    semantic.sum(axis=0)-total_form(reduced, q))))
                assert reduced_gate < 1e-7, (
                    coding, seed, name, reduced_gate)
                outside = semantic.reshape(6, -1) @ (
                    np.eye(physical_span.shape[0]) - physical_span)
                cancellation_ratio = float(
                    np.linalg.norm(outside.sum(axis=0)) /
                    np.sum(np.linalg.norm(outside, axis=1)))
                top = np.argsort(np.sum(coefficients**2, axis=(0, 1)))[-3:]
                for index in top:
                    item = semantic[index]
                    vec = item.ravel()
                    norm = np.linalg.norm(vec)
                    cosines = np.abs(physical_flat @ vec) / (
                        physical_norms * norm)
                    capture = np.linalg.norm(physical_span @ vec) / norm
                    cell_energy = np.sum(item**2, axis=(-1, -2))
                    rows.append(dict(coding=coding, seed=seed, gauge=name,
                        semantic_node=list(semantic_labels[index]),
                        top_three_rank=int(np.flatnonzero(top == index)[0]),
                        max_abs_physical_pair_cosine=float(np.max(cosines)),
                        best_physical_pair=list(physical_labels[
                            int(np.argmax(cosines))]),
                        physical_pair_span_capture=float(capture),
                        physical_pair_span_residual=float(
                            np.sqrt(max(0., 1-capture*capture))),
                        semantic_max_cell_energy_fraction=float(
                            np.max(cell_energy)/np.sum(cell_energy)),
                        mean_physical_max_cell_energy_fraction=float(
                            np.mean(physical_cell_purity)),
                        native_form_max_error=native_gate,
                        reduced_form_max_error=reduced_gate,
                        full_vs_reduced_form_relative_error=float(
                            np.linalg.norm(semantic.sum(axis=0)-original) /
                            np.linalg.norm(original)),
                        outside_span_cancellation_ratio_all_six=
                            cancellation_ratio))
            print(coding, seed, 'done', flush=True)
    summary = {}
    for coding, _ in CODINGS:
        for gauge in GAUGES:
            part = [r for r in rows if r['coding'] == coding and
                    r['gauge'] == gauge]
            assert len(part) == 30
            summary[f'{coding}__{gauge}'] = {
                field: float(np.mean([r[field] for r in part]))
                for field in ('max_abs_physical_pair_cosine',
                              'physical_pair_span_capture',
                              'physical_pair_span_residual',
                              'semantic_max_cell_energy_fraction',
                              'mean_physical_max_cell_energy_fraction',
                              'full_vs_reduced_form_relative_error',
                              'outside_span_cancellation_ratio_all_six')}
    result = dict(description='Exact polynomial coefficient comparison of three strongest function-level gauge nodes versus six physical attention-head-pair contributions in the learned quadratic readout. Original and reduced node sums checked exactly against their respective full forms. No task labels or sampled inputs enter the matching.', codings=[c for c, _ in CODINGS],
                  gauges=GAUGES, n_seeds=10, rows=rows, summary=summary)
    path = HERE / 'shared_pair_physical_node_alignment.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    for name, item in summary.items():
        print(name, {k:round(v, 3) for k,v in item.items()}, flush=True)
    return result


if __name__ == '__main__':
    run()

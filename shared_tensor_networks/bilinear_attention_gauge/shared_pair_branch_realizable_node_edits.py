"""Fit feasible six-branch QK edits to semantic-node deletion effects.

Each of three trained attention heads has two QK scores. A branch dose in
[0,1] replaces its score s by (1-dose)*s + dose*1, exactly as in the
existing branch-intervention experiments. The six doses are fitted on
seen-combination sentences; rare combinations and intermediate edit
strengths are independent tests.
"""

import json

import numpy as np
import torch
from scipy.optimize import minimize

from analyze_three_count_language import flatten
from analyze_toy_language import rms
from dual_joint_function_canonical import HERE
from shared_pair_data_equivalent_node_edits import sample, evaluate_forms
from shared_pair_physical_node_alignment import semantic_forms
from shared_pair_branch_interventions import branch_tables
from train_shared_pair_count import SharedPairCountModel
from train_shared_pair_dense import examples_dense
from train_three_count_language import examples as examples_paired


N_FIT = 10000
N_TEST = 20000
CODINGS = (('paired', 'shared_pair_count', examples_paired),
           ('dense', 'shared_pair_dense', examples_dense))


def head_branch_tables(model):
    """Base, qk1, qk2, and interaction tables, all [head, query, bit, key]."""
    base = model.backbone
    embedding = base.embed.weight.detach().numpy()
    query, key = embedding[8:10], embedding[:8]
    width = base.qk_width
    original, first, second, cross = [], [], [], []
    for head in range(base.heads):
        sl = slice(head * width, (head + 1) * width)

        def score(qname, kname):
            q = rms(query @ getattr(base, qname).weight.detach().numpy()[sl].T)
            k = rms(key @ getattr(base, kname).weight.detach().numpy()[sl].T)
            return q @ k.T / width

        s1, s2 = score('q1', 'k1'), score('q2', 'k2')
        value = key @ base.v.weight.detach().numpy()[head]
        read = base.read.weight.detach().numpy()[:, head]
        factor = value[None, :, None] * read[None, None, :]

        def table(s):
            return np.moveaxis(s[:, :, None] * factor, -1, 1)

        original.append(table(s1 * s2))
        first.append(table((1-s1)*s2))
        second.append(table(s1*(1-s2)))
        cross.append(table((1-s1)*(1-s2)))
    return tuple(np.stack(group) for group in
                 (original, first, second, cross))


def evaluated_tables(tables, counts, query):
    """Return four arrays [sentence, head, bit]."""
    result = []
    for table in tables:
        out = np.empty((len(counts), 3, 3))
        for q in (0, 1):
            mask = query == q
            out[mask] = np.einsum('nk,hbk->nhb', counts[mask], table[:, q])
        result.append(out)
    return tuple(result)


def response(doses, evaluated, bias, q):
    """Exact bit and parity changes from six simultaneous QK score edits."""
    original, first, second, cross = evaluated
    first_dose = doses[::2]
    second_dose = doses[1::2]
    baseline = original.sum(axis=1) + bias
    added = (np.einsum('nhb,h->nb', first, first_dose) +
             np.einsum('nhb,h->nb', second, second_dose) +
             np.einsum('nhb,h->nb', cross, first_dose * second_dose))
    edited = baseline + added
    parity_delta = (np.einsum('ni,pij,nj->np', edited, q, edited) -
                    np.einsum('ni,pij,nj->np', baseline, q, baseline))
    return added, parity_delta


def fit_doses(evaluated, target, bias, q):
    # Deterministic starts help distinguish a local basin from a missing edit.
    starts = [np.zeros(6), np.ones(6), np.full(6, .5)]
    starts += [np.eye(6)[i] for i in range(6)]

    def objective(doses):
        _, delta = response(doses, evaluated, bias, q)
        return float(np.mean((delta + target)**2))

    fits = [minimize(objective, start, bounds=[(0., 1.)]*6,
                     method='L-BFGS-B', options={'maxiter': 180,
                                                'ftol': 1e-11})
            for start in starts]
    best = min(fits, key=lambda result: result.fun)
    return best.x, float(best.fun)


def scores(doses, evaluated, target, bias, q, rare, strength):
    bits, parity = response(strength*doses, evaluated, bias, q)
    residual = parity + strength*target
    full_energy = max(float(np.mean((strength*target)**2)), 1e-20)
    rare_energy = max(float(np.mean((strength*target[rare])**2)), 1e-20)
    return dict(relative_parity_mse=float(np.mean(residual**2)/full_energy),
        seen_relative_parity_mse=float(np.mean(residual[~rare]**2) /
            max(float(np.mean((strength*target[~rare])**2)), 1e-20)),
        rare_relative_parity_mse=float(np.mean(residual[rare]**2)/rare_energy),
        rare_mse_over_full_target_energy=float(
            np.mean(residual[rare]**2)/full_energy),
        bit_change_rms=float(np.sqrt(np.mean(bits**2))),
        parity_target_rms=float(np.sqrt(full_energy)))


def run():
    torch.set_num_threads(1)
    rows = []
    for coding, prefix, source in CODINGS:
        analysis = json.loads((HERE / f'{prefix}_analysis.json').read_text())
        gauge = json.loads((HERE / ('shared_pair_gauge.json' if coding ==
                    'paired' else 'shared_pair_dense_gauge.json')).read_text())
        for seed in range(10):
            model = SharedPairCountModel().double()
            model.load_state_dict(torch.load(HERE/f'{prefix}_seed{seed}.pt',
                                  map_location='cpu', weights_only=True))
            model.eval()
            record = next(row for row in analysis['rows'] if row['seed'] == seed)
            q = np.asarray(record['quadratic_effective'])
            matrix = np.asarray(record['matrices']['uniform_rank3'])
            gauge_row = next(row for row in gauge['rows'] if row['seed'] == seed
                             and row['reduction'] == 'uniform_rank3')
            dictionary = np.asarray(
                gauge_row['gauges']['joint_nullspace']['dictionary'])
            semantic, labels, coefficients = semantic_forms(matrix, q,
                                                              dictionary)
            selected = np.argsort(np.sum(coefficients**2, axis=(0, 1)))[-3:]
            tables = head_branch_tables(model)
            # Sum of head tables must reproduce the exact unedited token map.
            from analyze_three_count_language import head_tables
            native = np.stack([flatten(t).reshape(2, 3, 8) for t in
                               head_tables(model.backbone)])
            assert np.max(np.abs(tables[0]-native)) < 1e-9
            # Cross-check both individually edited branches and their
            # interaction against the independent exact-table implementation.
            reference = branch_tables(model, .37)
            original, first, second, cross = tables
            expected = {
                'original': original,
                'qk1_constant': original + .37*first,
                'qk2_constant': original + .37*second,
                'both_constant': original + .37*(first+second)+.37**2*cross}
            for condition, per_head in expected.items():
                table = flatten(np.moveaxis(per_head.sum(axis=0), 1, 2))
                assert np.max(np.abs(table-reference[condition])) < 1e-9
            fit_counts, fit_query, _ = sample(source, N_FIT,
                                               810000+seed, True)
            test_counts, test_query, attrs = sample(source, N_TEST,
                                                    820000+seed, False)
            assert int(attrs['heldout'].sum()) > 100
            fit_evaluated = evaluated_tables(tables, fit_counts, fit_query)
            test_evaluated = evaluated_tables(tables, test_counts, test_query)
            fit_targets = evaluate_forms(semantic, fit_counts, fit_query)
            test_targets = evaluate_forms(semantic, test_counts, test_query)
            bias = model.backbone.read.bias.detach().numpy()
            for index in selected:
                target_fit, target_test = (fit_targets[:, index],
                                            test_targets[:, index])
                doses, fit_mse = fit_doses(fit_evaluated, target_fit, bias, q)
                measures = {str(strength): scores(doses, test_evaluated,
                    target_test, bias, q, attrs['heldout'], strength)
                    for strength in (.25, .5, 1.)}
                rows.append(dict(coding=coding, seed=seed,
                    semantic_node=list(labels[index]),
                    fitted_branch_doses=doses.tolist(), fit_mse=fit_mse,
                    n_fit=N_FIT, n_test=N_TEST,
                    n_rare=int(attrs['heldout'].sum()), scores=measures))
            print(coding, seed, flush=True)
    summary = {}
    for coding, _, _ in CODINGS:
        for node in ((0, 1), (0, 2), (1, 2)):
            part = [row for row in rows if row['coding'] == coding and
                    tuple(row['semantic_node']) == node]
            assert len(part) == 10
            summary[f'{coding}__{node}'] = {strength: {
                field: float(np.mean([row['scores'][strength][field]
                                      for row in part])) for field in
                ('relative_parity_mse', 'rare_mse_over_full_target_energy',
                 'bit_change_rms', 'parity_target_rms')}
                for strength in ('0.25', '0.5', '1.0')}
    result = dict(description='Fit six feasible QK-branch-to-constant doses on seen task combinations to imitate each joint-nullspace semantic-node deletion. Test on fresh sentences, rare whole-combination holdout, and fractionally scaled unseen edit strengths. Exact trained-model branch tables and quadratic readout; fit uses parity effects but no answer labels.',
        n_fit=N_FIT, n_test=N_TEST, n_models=20, rows=rows, summary=summary)
    path = HERE/'shared_pair_branch_realizable_node_edits.json'
    path.write_text(json.dumps(result, indent=2)+'\n')
    for name, value in summary.items():
        print(name, {dose: {k:round(v,3) for k,v in s.items()}
                     for dose,s in value.items()}, flush=True)
    return result


if __name__ == '__main__':
    run()

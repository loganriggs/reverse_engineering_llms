"""Fit an exact rank-three gauge to finite XOR responses at one dose.

Fit contexts and task outputs are disjoint from final test contexts/outputs.
This is a coordinate intervention in the reduced function, not a physical
attention-head intervention.
"""

import json

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment, minimize
from scipy.spatial.transform import Rotation

from diverse_task_benchmark import DiverseTaskModel, exact_bits
from diverse_task_gauge import fit as fit_rotation, gradients, sparsity
from dual_joint_function_canonical import HERE
from three_count_task_fit import count_features
from train_shared_pair_dense import TOKEN_SIGNS, examples_dense


FIT_TASKS = (3, 4, 6)
HELD_TASKS = (0, 1, 2, 5, 7, 8, 9, 10)
FIT_DOSE = .25


def readout_numpy(model, bits):
    first, last = model.readout[0], model.readout[2]
    w1, b1 = first.weight.detach().numpy(), first.bias.detach().numpy()
    w2, b2 = last.weight.detach().numpy(), last.bias.detach().numpy()
    hidden = np.tanh(np.tanh(bits / 4) @ w1.T + b1)
    return np.concatenate((bits, hidden @ w2.T + b2), axis=-1)


def panel(model, cut, bias, seed, n, holdout):
    x, _, _ = examples_dense(n, seed, holdout)
    bits = exact_bits(cut, bias, x).numpy()
    return {
        'tokens': x,
        'bits': bits,
        'base': readout_numpy(model, bits),
        'counts': count_features(x.numpy()).astype(np.float64),
        'query': x[:, -1].numpy() - 8,
    }


def finite_delta(model, cut, dictionary, data, dose):
    coeff = cut.reshape(2, 3, 8) @ dictionary.T
    coords = data['counts'] @ dictionary.T
    contributions = np.einsum('noi,ni->nio', coeff[data['query']], coords)
    rms = np.sqrt(np.mean(contributions * contributions, axis=(0, 2)))
    # Match RMS bit-logit dose across axes. The mean axis RMS changes by
    # less than .1% among gauges in this experiment; record it separately.
    edit = dose * (rms.mean() / rms)[None, :, None] * contributions
    output = readout_numpy(model, (data['bits'][:, None, :] - edit).reshape(-1, 3))
    output = output.reshape(len(data['bits']), 3, -1)
    return output - data['base'][:, None, :], rms


def score(model, cut, dictionary, data, dose, tasks):
    delta, rms = finite_delta(model, cut, dictionary, data, dose)
    selected = delta[:, :, tasks]
    return {
        'sparsity_l1_l2': sparsity(selected.transpose(0, 2, 1)),
        'response_rms': float(np.sqrt(np.mean(selected * selected))),
        'response_mean_abs': float(np.abs(selected).mean()),
        'mean_axis_bit_rms': float(rms.mean()),
    }


def fit(model, cut, base, train, validation, seed):
    rng = np.random.default_rng(seed)
    starts = np.vstack((np.zeros(3), rng.uniform(-np.pi, np.pi, (64, 3))))
    def dictionary(angles):
        return Rotation.from_euler('xyz', angles).as_matrix() @ base
    def objective(angles):
        return score(model, cut, dictionary(angles), train, FIT_DOSE,
                     FIT_TASKS)['sparsity_l1_l2']
    values = np.array([objective(x) for x in starts])
    candidates = []
    for idx in np.argsort(values)[:4]:
        result = minimize(objective, starts[idx], method='Powell',
                          options={'maxiter': 100, 'xtol': 1e-5, 'ftol': 1e-7})
        candidate = dictionary(result.x)
        candidates.append((score(model, cut, candidate, validation,
                                 FIT_DOSE, FIT_TASKS)['sparsity_l1_l2'],
                           candidate, float(result.fun)))
    valid, selected, fit_score = min(candidates, key=lambda x: x[0])
    return selected, {'fit_score': fit_score, 'validation_score': valid,
                      'n_starts': len(starts), 'n_local_refinements': len(candidates)}


def activation_weighted_gradient_fit(model, cut, bias, base, train, seed):
    jac = gradients(model, train['tokens'], cut, bias)
    coeff = cut.reshape(2, 3, 8) @ base.T
    grad = np.einsum('nto,nok->ntk', jac, coeff[train['query']])
    coords = train['counts'] @ base.T
    def objective(dictionary):
        rotation = dictionary @ base.T
        effect = (grad @ rotation.T) * (coords @ rotation.T)[:, None, :]
        return sparsity(effect[:, FIT_TASKS])
    return fit_rotation(base, objective, seed)


def run(seeds=range(5)):
    torch.set_num_threads(1)
    prior = json.loads((HERE / 'diverse_task_gauge_dense_xor.json').read_text())
    assert prior['fit_tasks'] == ['A_xor_B', 'B_xor_C', 'A_xor_C']
    truth = TOKEN_SIGNS.T / np.sqrt(8)
    rows = []
    for seed in seeds:
        model = DiverseTaskModel().double()
        model.load_state_dict(torch.load(HERE / f'diverse_task_benchmark_dense_seed{seed}.pt',
                                         map_location='cpu', weights_only=True))
        model.eval()
        benchmark = json.loads((HERE / f'diverse_task_benchmark_dense_seed{seed}.json').read_text())
        cut = np.asarray(benchmark['matrices']['uniform_r3'])
        bias = model.backbone.read.bias.detach().numpy()
        base = np.asarray(prior['rows'][seed]['gauges']['svd']['dictionary'])
        train = panel(model, cut, bias, 881000 + seed, 600, True)
        validation = panel(model, cut, bias, 882000 + seed, 600, True)
        test = panel(model, cut, bias, 883000 + seed, 3000, False)
        fitted, fit_info = fit(model, cut, base, train, validation, 884000 + seed)
        activation = activation_weighted_gradient_fit(model, cut, bias, base,
                                                     train, 885000 + seed)
        choices = {name: np.asarray(prior['rows'][seed]['gauges'][name]['dictionary'])
                   for name in ('svd', 'key_l1', 'fit_response_l1')}
        choices['fit_finite_l1'] = fitted
        choices['fit_activation_gradient_l1'] = activation
        gauges = {}
        for name, dictionary in choices.items():
            match = np.abs(dictionary @ truth.T)
            axis, family = linear_sum_assignment(-match)
            item = {'known_variable_matched_cosine': float(match[axis, family].mean()),
                    'exact_function_relative_error': float(np.linalg.norm(
                        cut @ dictionary.T @ dictionary - cut) / np.linalg.norm(cut)),
                    'dictionary': dictionary.tolist()}
            assert item['exact_function_relative_error'] < 1e-10
            for dose in (.05, .25, 1.0):
                item[f'held_dose{dose}'] = score(model, cut, dictionary, test,
                                                 dose, HELD_TASKS)
                item[f'fit_dose{dose}'] = score(model, cut, dictionary, test,
                                                dose, FIT_TASKS)
            gauges[name] = item
        rows.append({'seed': seed, 'fit_info': fit_info, 'gauges': gauges})
        print(seed, {name: round(item['held_dose1.0']['sparsity_l1_l2'], 2)
                     for name, item in gauges.items()}, flush=True)
    result = {'fit_dose': FIT_DOSE, 'fit_tasks': list(FIT_TASKS),
              'held_tasks': list(HELD_TASKS), 'fit_contexts_per_seed': 600,
              'validation_contexts_per_seed': 600, 'test_contexts_per_seed': 3000,
              'rows': rows}
    (HERE / 'diverse_task_finite_fit_dense_xor.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    run()

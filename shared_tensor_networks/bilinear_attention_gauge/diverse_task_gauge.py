"""Compare exact rank-three gauges using weight-only and task-response priors."""

import json

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment, minimize
from scipy.spatial.transform import Rotation

from analyze_three_count_language import flatten, head_tables
from diverse_task_benchmark import DiverseTaskModel, FIT_TASKS, HELD_TASKS, TASKS, exact_bits, rank_cut
from dual_joint_function_canonical import HERE
from three_count_nonlinear_gauge import TRUTH
from train_shared_pair_dense import TOKEN_SIGNS, examples_dense
from train_three_count_language import examples


def gradients(model, tokens, matrix, bias):
    """Per-context Jacobian of all task logits with respect to three bit logits."""
    bits = exact_bits(matrix, bias, tokens).detach()
    bits.requires_grad_(True)
    out = model.from_bits(bits)
    return np.stack([torch.autograd.grad(out[:, task].sum(), bits,
                                         retain_graph=task < len(TASKS) - 1)[0].detach().numpy()
                     for task in range(len(TASKS))], axis=1)


def sensitivity(model, matrix, bias, base, tokens):
    jacobian = gradients(model, tokens, matrix, bias)
    c = matrix.reshape(2, 3, 8) @ base.T
    query = tokens[:, -1].numpy().astype(int) - 8
    return np.einsum('nto,nok->ntk', jacobian, c[query])


def sparsity(grad):
    # Invariant total L2 is fixed by the orthogonal gauge. Lower is sparser.
    return float(np.abs(grad).sum() / np.maximum(np.sqrt(np.sum(grad * grad)), 1e-20))


def fit(base, objective, seed):
    rng = np.random.default_rng(seed)
    starts = np.vstack((np.zeros(3), rng.uniform(-np.pi, np.pi, (192, 3))))
    def score(angles):
        return objective(Rotation.from_euler('xyz', angles).as_matrix() @ base)
    values = np.array([score(x) for x in starts])
    best = (float('inf'), None)
    for i in np.argsort(values)[:8]:
        result = minimize(score, starts[i], method='Powell',
                          options={'maxiter': 150, 'xtol': 1e-7, 'ftol': 1e-9})
        if result.fun < best[0]:
            best = (float(result.fun), result.x)
    assert best[1] is not None
    return Rotation.from_euler('xyz', best[1]).as_matrix() @ base


def profile(dictionary, matrix, base, train_grad, test_grad, truth, fit_tasks, held_tasks):
    rotation = dictionary @ base.T
    match = np.abs(dictionary @ truth.T)
    axis, family = linear_sum_assignment(-match)
    result = {
        'key_dictionary_l1': float(np.abs(dictionary).sum()),
        'known_variable_matched_cosine': float(match[axis, family].mean()),
        'fit_task_sensitivity_l1': sparsity(train_grad[:, fit_tasks] @ rotation.T),
        'held_task_sensitivity_l1': sparsity(test_grad[:, held_tasks] @ rotation.T),
        'fit_task_sensitivity_l1_per_entry': float(np.abs(train_grad[:, fit_tasks] @ rotation.T).mean()),
        'held_task_sensitivity_l1_per_entry': float(np.abs(test_grad[:, held_tasks] @ rotation.T).mean()),
    }
    # Test that this is a gauge change only, not a new reduced function.
    coefficients = matrix @ dictionary.T
    result['exact_function_relative_error'] = float(np.linalg.norm(coefficients @ dictionary - matrix) /
                                                    np.linalg.norm(matrix))
    return result


def run(seeds=range(5), coding='sparse', fit_mode='standard'):
    torch.set_num_threads(1)
    rows = []
    source = examples if coding == 'sparse' else examples_dense
    truth = TRUTH if coding == 'sparse' else TOKEN_SIGNS.T / np.sqrt(8)
    task_split = {
        'standard': (FIT_TASKS, HELD_TASKS),
        'composite': ((3, 4, 5, 6, 7), (0, 1, 2, 8, 9, 10)),
        'xor': ((3, 4, 6), (0, 1, 2, 5, 7, 8, 9, 10)),
    }
    fit_tasks, held_tasks = task_split[fit_mode]
    for seed in seeds:
        model = DiverseTaskModel().double()
        stem = f'diverse_task_benchmark_seed{seed}' if coding == 'sparse' else \
               f'diverse_task_benchmark_dense_seed{seed}'
        model.load_state_dict(torch.load(HERE / f'{stem}.pt',
                                         map_location='cpu', weights_only=True))
        model.eval()
        native = flatten(sum(head_tables(model.backbone)))
        matrix = rank_cut(native, None, 3)
        bias = model.backbone.read.bias.detach().numpy()
        _, s, vh = np.linalg.svd(matrix, full_matrices=False)
        assert s[3] / s[0] < 1e-10, s
        base = vh[:3]
        train_x, _, _ = source(3000, 861000 + seed, True)
        test_x, _, _ = source(3000, 862000 + seed, False)
        train_grad = sensitivity(model, matrix, bias, base, train_x)
        test_grad = sensitivity(model, matrix, bias, base, test_x)
        key = fit(base, lambda d: np.abs(d).sum(), 863000 + seed)
        response = fit(base,
                       lambda d: sparsity(train_grad[:, fit_tasks] @ (d @ base.T).T),
                       864000 + seed)
        choices = {'svd': base, 'key_l1': key, 'fit_response_l1': response}
        results = {name: profile(d, matrix, base, train_grad, test_grad,
                                 truth, fit_tasks, held_tasks)
                   for name, d in choices.items()}
        for name, result in results.items():
            assert result['exact_function_relative_error'] < 1e-10, (seed, name)
            result['dictionary'] = choices[name].tolist()
        random_rng = np.random.default_rng(865000 + seed)
        random_profiles = [profile(Rotation.random(random_state=random_rng).as_matrix() @ base,
                                   matrix, base, train_grad, test_grad,
                                   truth, fit_tasks, held_tasks)
                           for _ in range(128)]
        results['random_rotation_mean'] = {
            key: float(np.mean([p[key] for p in random_profiles]))
            for key in ('key_dictionary_l1', 'known_variable_matched_cosine',
                        'fit_task_sensitivity_l1', 'held_task_sensitivity_l1')}
        rows.append({'seed': seed,
                     'native_to_rank3_relative_table_error': float(np.linalg.norm(matrix - native) /
                                                                   np.linalg.norm(native)),
                     'gauges': results})
        print('seed', seed, {n: round(v['known_variable_matched_cosine'], 3)
                              for n, v in results.items()}, flush=True)
    output = {'description': 'Exact gauge changes of a fixed uniform rank-three cut; task-response prior sees only designated fit outputs',
              'coding': coding,
              'fit_mode': fit_mode,
              'fit_tasks': [TASKS[i] for i in fit_tasks],
              'held_tasks': [TASKS[i] for i in held_tasks],
              'rows': rows}
    filename = 'diverse_task_gauge.json' if coding == 'sparse' else 'diverse_task_gauge_dense.json'
    if fit_mode != 'standard':
        filename = filename.removesuffix('.json') + f'_{fit_mode}.json'
    (HERE / filename).write_text(json.dumps(output, indent=2) + '\n')
    return output


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--coding', choices=('sparse', 'dense'), default='sparse')
    parser.add_argument('--fit-mode', choices=('standard', 'composite', 'xor'), default='standard')
    args = parser.parse_args()
    run(coding=args.coding, fit_mode=args.fit_mode)

"""Hold-out one-attribute token edits for exact-gauge top-axis explanations."""

import json

import numpy as np
import torch

from analyze_three_count_language import flatten, head_tables
from diverse_task_benchmark import DiverseTaskModel, exact_bits
from diverse_task_finite_fit import HELD_TASKS, readout_numpy
from dual_joint_function_canonical import HERE
from three_count_task_fit import count_features
from train_shared_pair_dense import TOKEN_SIGNS, examples_dense


def model_outputs(model, matrix, bias, tokens):
    return readout_numpy(model, exact_bits(matrix, bias, tokens).numpy())


def run(seeds=range(5), contexts_per_pair=200):
    torch.set_num_threads(1)
    fit = json.loads((HERE / 'diverse_task_finite_fit_dense_xor.json').read_text())
    rows = []
    for seed in seeds:
        model = DiverseTaskModel().double()
        model.load_state_dict(torch.load(HERE / f'diverse_task_benchmark_dense_seed{seed}.pt',
                                         map_location='cpu', weights_only=True))
        model.eval()
        native = flatten(sum(head_tables(model.backbone)))
        cut = np.asarray(json.loads((HERE / f'diverse_task_benchmark_dense_seed{seed}.json').read_text())
                         ['matrices']['uniform_r3'])
        bias = model.backbone.read.bias.detach().numpy()
        source, _, _ = examples_dense(20000, 893000 + seed, False)
        rng = np.random.default_rng(894000 + seed)
        names = tuple(fit['rows'][seed]['gauges'])
        accum = {name: {'top1_sq': 0., 'top2_sq': 0., 'full_sq': 0.,
                        'native_sq': 0., 'valid_top1_sq': 0.,
                        'valid_native_sq': 0., 'n': 0} for name in names}
        direct_gate = 0.
        pairs = []
        for k in range(8):
            for bit in range(3):
                replacement = k ^ (1 << bit)
                eligible = torch.where((source[:, :-1] == k).any(1))[0].numpy()
                selected = rng.choice(eligible, size=contexts_per_pair, replace=False)
                before = source[selected].clone()
                after = before.clone()
                first = torch.argmax((after[:, :-1] == k).to(torch.int64), dim=1)
                after[torch.arange(len(after)), first] = replacement
                count = count_features(before.numpy()).astype(np.float64)
                edited_count = count.copy()
                edited_count[:, k] -= 1
                edited_count[:, replacement] += 1
                valid_edit = np.all((edited_count @ TOKEN_SIGNS) != 0, axis=1)
                query = before[:, -1].numpy() - 8
                native_delta = model_outputs(model, native, bias, after) - \
                               model_outputs(model, native, bias, before)
                cut_before_bits = exact_bits(cut, bias, before).numpy()
                cut_after_bits = exact_bits(cut, bias, after).numpy()
                cut_base = readout_numpy(model, cut_before_bits)
                cut_delta = readout_numpy(model, cut_after_bits) - cut_base
                # The table algebra must match direct model forwards on both
                # original and replaced discrete-token sequences.
                if bit == 0 and k == 0:
                    with torch.no_grad():
                        direct_gate = max(direct_gate,
                            float(np.max(np.abs(model(before[:32]).numpy() -
                                                model_outputs(model, native, bias, before[:32])))),
                            float(np.max(np.abs(model(after[:32]).numpy() -
                                                model_outputs(model, native, bias, after[:32])))))
                pair = {'key': k, 'replacement': replacement,
                        'attribute_bit': bit, 'n': len(before),
                        'edited_nonzero_contrast_fraction': float(np.mean(valid_edit)),
                        'gauges': {}}
                for name in names:
                    dictionary = np.asarray(fit['rows'][seed]['gauges'][name]['dictionary'])
                    coeff = cut.reshape(2, 3, 8) @ dictionary.T
                    dz = dictionary[:, replacement] - dictionary[:, k]
                    axis_effect = coeff[query] * dz[None, None, :]
                    size = np.sum(axis_effect * axis_effect, axis=1)
                    order = np.argsort(-size, axis=1)
                    top1 = np.take_along_axis(axis_effect, order[:, None, :1], axis=2).sum(2)
                    top2 = np.take_along_axis(axis_effect, order[:, None, :2], axis=2).sum(2)
                    pred1 = readout_numpy(model, cut_before_bits + top1) - cut_base
                    pred2 = readout_numpy(model, cut_before_bits + top2) - cut_base
                    reference = cut_delta[:, HELD_TASKS]
                    a = accum[name]
                    a['top1_sq'] += float(np.sum((pred1[:, HELD_TASKS] - reference)**2))
                    a['top2_sq'] += float(np.sum((pred2[:, HELD_TASKS] - reference)**2))
                    a['full_sq'] += float(np.sum((cut_delta[:, HELD_TASKS] -
                                                 native_delta[:, HELD_TASKS])**2))
                    a['native_sq'] += float(np.sum(native_delta[:, HELD_TASKS]**2))
                    a['valid_top1_sq'] += float(np.sum((pred1[valid_edit][:, HELD_TASKS] -
                                                       reference[valid_edit])**2))
                    a['valid_native_sq'] += float(np.sum(native_delta[valid_edit][:, HELD_TASKS]**2))
                    a['n'] += len(before)
                    pair['gauges'][name] = {
                        'selected_axis_counts': np.bincount(order[:, 0], minlength=3).tolist(),
                        'top1_error_over_native_effect_energy': float(np.sum(
                            (pred1[:, HELD_TASKS] - reference)**2) /
                            max(np.sum(native_delta[:, HELD_TASKS]**2), 1e-20)),
                        'full_cut_error_over_native_effect_energy': float(np.sum(
                            (cut_delta[:, HELD_TASKS] - native_delta[:, HELD_TASKS])**2) /
                            max(np.sum(native_delta[:, HELD_TASKS]**2), 1e-20)),
                    }
                pairs.append(pair)
        assert direct_gate < 1e-8, direct_gate
        scores = {name: {
            'top1_vs_cut_error_over_native_effect_energy': a['top1_sq'] / max(a['native_sq'], 1e-20),
            'top2_vs_cut_error_over_native_effect_energy': a['top2_sq'] / max(a['native_sq'], 1e-20),
            'full_cut_relative_to_native_mse': a['full_sq'] / max(a['native_sq'], 1e-20),
            'valid_edit_top1_error_over_native_effect_energy': a['valid_top1_sq'] /
                                                                max(a['valid_native_sq'], 1e-20),
            'n_context_edits': a['n'],
        } for name, a in accum.items()}
        rows.append({'seed': seed, 'exact_native_forward_gate': direct_gate,
                     'pairs': pairs, 'scores': scores})
        print(seed, {name: round(score['top1_vs_cut_error_over_native_effect_energy'], 4)
                     for name, score in scores.items()}, flush=True)
    result = {'description': '24 directed one-bit key-token substitutions per seed, unseen edit family',
              'contexts_per_pair': contexts_per_pair,
              'held_task_indices': list(HELD_TASKS), 'rows': rows}
    (HERE / 'diverse_task_token_edits_dense_xor.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    run()

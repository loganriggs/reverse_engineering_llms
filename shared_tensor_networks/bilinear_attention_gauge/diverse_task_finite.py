"""Fresh finite-axis response check for dense-code XOR-fitted gauges."""

import json

import numpy as np
import torch

from diverse_task_benchmark import DiverseTaskModel, exact_bits
from diverse_task_gauge import sparsity
from dual_joint_function_canonical import HERE
from three_count_task_fit import count_features
from train_shared_pair_dense import examples_dense


def run(seeds=range(5), ntest=3000):
    torch.set_num_threads(1)
    gauge = json.loads((HERE / 'diverse_task_gauge_dense_xor.json').read_text())
    assert gauge['fit_tasks'] == ['A_xor_B', 'B_xor_C', 'A_xor_C']
    held = [0, 1, 2, 5, 7, 8, 9, 10]
    rows = []
    for seed in seeds:
        model = DiverseTaskModel().double()
        model.load_state_dict(torch.load(HERE / f'diverse_task_benchmark_dense_seed{seed}.pt',
                                         map_location='cpu', weights_only=True))
        model.eval()
        x, _, _ = examples_dense(ntest, 872000 + seed, False)
        cut = np.asarray(json.loads((HERE / f'diverse_task_benchmark_dense_seed{seed}.json').read_text())
                         ['matrices']['uniform_r3'])
        bias = model.backbone.read.bias.detach().numpy()
        bits = exact_bits(cut, bias, x).numpy()
        counts = count_features(x.numpy()).astype(np.float64)
        query = x[:, -1].numpy() - 8
        with torch.no_grad():
            base = model.from_bits(torch.from_numpy(bits)).numpy()
        results = {}
        for name in ('svd', 'key_l1', 'fit_response_l1'):
            dictionary = np.asarray(gauge['rows'][seed]['gauges'][name]['dictionary'])
            coeff = cut.reshape(2, 3, 8) @ dictionary.T
            coords = counts @ dictionary.T
            contributions = np.einsum('noi,ni->nio', coeff[query], coords)
            # Each axis has the same aggregate RMS bit-logit perturbation.
            rms = np.sqrt(np.mean(contributions * contributions, axis=(0, 2)))
            dose_factor = rms.mean() / rms
            result = {'axis_bit_rms': rms.tolist(), 'dose_factor': dose_factor.tolist()}
            for fraction in (.05, .25, 1.0):
                edited = bits[:, None, :] - fraction * dose_factor[None, :, None] * contributions
                with torch.no_grad():
                    output = model.from_bits(torch.from_numpy(edited.reshape(-1, 3)))
                    output = output.numpy().reshape(len(x), 3, -1)
                delta = output[:, :, held] - base[:, None, held]
                # Axis dimension is last for the same L1/L2 convention as the gradient test.
                result[f'held_finite_l1_dose{fraction}'] = sparsity(delta.transpose(0, 2, 1))
                result[f'held_finite_mean_abs_dose{fraction}'] = float(np.abs(delta).mean())
            results[name] = result
        rows.append({'seed': seed, 'gauges': results})
        print(seed, {name: round(v['held_finite_l1_dose1.0'], 2)
                     for name, v in results.items()}, flush=True)
    output = {'description': 'Fresh dense-token contexts, XOR-fitted exact gauges, held-output finite coordinate deletion',
              'held_tasks': [gauge['held_tasks'][i] for i in range(len(held))],
              'ntest_per_seed': ntest, 'rows': rows}
    (HERE / 'diverse_task_finite_dense_xor.json').write_text(json.dumps(output, indent=2) + '\n')
    return output


if __name__ == '__main__':
    run()

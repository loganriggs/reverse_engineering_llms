"""Train a multi-rule attention toy and test whole-task transfer of rank cuts.

The first-layer table is exact for this architecture. The supervised cut sees
only six output channels; the other five are never used for cut fitting or
selection. This tests transfer of a reduction, not discovery of a new task.
"""

import argparse
import copy
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from analyze_three_count_language import flatten, head_tables
from dual_joint_function_canonical import HERE
from three_count_metric import pair_weights, weighted_fit
from three_count_task_fit import count_features
from train_shared_pair_dense import examples_dense
from train_three_count_language import ThreeCountAttention, examples


TASKS = ('A', 'B', 'C', 'A_xor_B', 'B_xor_C', 'A_and_B',
         'A_xor_C', 'A_and_C', 'majority', 'exactly_one', 'all_equal')
FIT_TASKS = tuple(range(6))
HELD_TASKS = tuple(range(6, len(TASKS)))


def labels(bits):
    a, b, c = (bits[:, i].bool() for i in range(3))
    count = bits.sum(1)
    return torch.stack((a, b, c, a ^ b, b ^ c, a & b,
                        a ^ c, a & c, count >= 2, count == 1,
                        (a == b) & (b == c)), 1).float()


class DiverseTaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ThreeCountAttention()
        self.readout = nn.Sequential(nn.Linear(3, 48), nn.Tanh(),
                                     nn.Linear(48, len(TASKS) - 3))

    def from_bits(self, bits):
        return torch.cat((bits, self.readout(torch.tanh(bits / 4))), 1)

    def forward(self, tokens):
        return self.from_bits(self.backbone(tokens))


def exact_bits(matrix, bias, tokens):
    counts = count_features(tokens.numpy()).astype(np.float64)
    query = tokens[:, -1].numpy().astype(int) - 8
    return torch.from_numpy(bias + np.einsum('nk,nok->no', counts,
                                            matrix.reshape(2, 3, 8)[query]))


def evaluate(model, x, y, attrs, matrix=None, bias=None):
    model.eval()
    with torch.no_grad():
        out = model(x) if matrix is None else model.from_bits(exact_bits(matrix, bias, x))
        pred = out > 0
        correct = (pred == y.bool()).numpy()
        bce = F.binary_cross_entropy_with_logits(out, y, reduction='none').numpy()
    result = {}
    for name, mask in (('all', np.ones(len(x), dtype=bool)),
                       ('rare_heldout', attrs['heldout'])):
        result[name] = {
            'n': int(mask.sum()),
            'task_accuracy': dict(zip(TASKS, correct[mask].mean(0).tolist())),
            'task_bce': dict(zip(TASKS, bce[mask].mean(0).tolist())),
            'fit_joint_accuracy': float(correct[mask][:, FIT_TASKS].all(1).mean()),
            'held_joint_accuracy': float(correct[mask][:, HELD_TASKS].all(1).mean()),
        }
    return result


def rank_cut(matrix, weights, rank):
    if weights is None:
        u, s, vh = np.linalg.svd(matrix, full_matrices=False)
        return (u[:, :rank] * s[:rank]) @ vh[:rank]
    return weighted_fit(matrix, weights, rank)[0]


def task_cut(model, native, bias, xtrain, xval, rank, init, steps=500):
    """Distill only FIT_TASKS of the frozen teacher, select on those tasks."""
    with torch.no_grad():
        teacher_train = model(xtrain)[:, FIT_TASKS]
        teacher_val = model(xval)[:, FIT_TASKS]
    u, s, vh = np.linalg.svd(init, full_matrices=False)
    left = nn.Parameter(torch.tensor(u[:, :rank] * np.sqrt(s[:rank]), dtype=torch.float64))
    right = nn.Parameter(torch.tensor(np.sqrt(s[:rank])[:, None] * vh[:rank], dtype=torch.float64))
    optimizer = torch.optim.Adam((left, right), lr=.015)
    counts_train = torch.from_numpy(count_features(xtrain.numpy()).astype(np.float64))
    counts_val = torch.from_numpy(count_features(xval.numpy()).astype(np.float64))
    query_train = xtrain[:, -1] - 8
    query_val = xval[:, -1] - 8
    bias_t = torch.tensor(bias, dtype=torch.float64)

    def predict(counts, query):
        matrix = (left @ right).reshape(2, 3, 8)
        bits = bias_t + torch.einsum('nk,nok->no', counts, matrix[query])
        return model.from_bits(bits)[:, FIT_TASKS]

    rng = np.random.default_rng(92901 + rank)
    best = (float('inf'), None, 0)
    for step in range(steps + 1):
        if step % 25 == 0:
            with torch.no_grad():
                val = F.mse_loss(predict(counts_val, query_val), teacher_val).item()
            if val < best[0]:
                best = (val, (left @ right).detach().numpy().copy(), step)
        if step == steps:
            break
        idx = torch.from_numpy(rng.integers(0, len(xtrain), 1024))
        output = predict(counts_train[idx], query_train[idx])
        loss = F.mse_loss(output, teacher_train[idx])
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_((left, right), 5.)
        optimizer.step()
    return best[1], {'fit_validation_mse': best[0], 'selected_step': best[2]}


def run(seed, train_steps=2500, cut_steps=500, analysis_only=False, coding='sparse'):
    torch.set_num_threads(1)
    torch.manual_seed(850000 + seed)
    source = examples if coding == 'sparse' else examples_dense
    stem = HERE / f'diverse_task_benchmark_{coding}_seed{seed}'
    if coding == 'sparse':
        stem = HERE / f'diverse_task_benchmark_seed{seed}'
    train_x, train_bits, _ = source(50000, 851000 + seed, True)
    val_x, val_bits, val_attrs = source(10000, 852000 + seed, True)
    test_x, test_bits, test_attrs = source(20000, 853000 + seed, False)
    train_y, val_y, test_y = map(labels, (train_bits, val_bits, test_bits))
    assert not val_attrs['heldout'].any() and test_attrs['heldout'].sum() > 100
    model = DiverseTaskModel().double()
    if analysis_only:
        model.load_state_dict(torch.load(str(stem) + '.pt',
                                         map_location='cpu', weights_only=True))
        prior = json.loads((HERE / f'{stem.name}.json').read_text())
        selected_step = prior['model_selected_step']
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=1e-4)
        rng = np.random.default_rng(854000 + seed)
        best = (float('inf'), None, 0)
        for step in range(1, train_steps + 1):
            idx = torch.from_numpy(rng.integers(0, len(train_x), 512))
            output = model(train_x[idx])
            loss = F.binary_cross_entropy_with_logits(output[:, :3], train_y[idx, :3]) + \
                   F.binary_cross_entropy_with_logits(output[:, 3:], train_y[idx, 3:])
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            if step % 250 == 0 or step == train_steps:
                with torch.no_grad():
                    val = F.binary_cross_entropy_with_logits(model(val_x), val_y).item()
                if val < best[0]:
                    best = (val, copy.deepcopy(model.state_dict()), step)
                print('train', seed, step, round(val, 5), flush=True)
        model.load_state_dict(best[1])
        selected_step = best[2]
    model.eval()
    matrix = flatten(sum(head_tables(model.backbone)))
    bias = model.backbone.read.bias.detach().numpy()
    with torch.no_grad():
        gate = float((model(test_x[:128]) - model.from_bits(exact_bits(matrix, bias, test_x[:128]))).abs().max())
    assert gate < 1e-8, gate
    weights = pair_weights(train_x.numpy())
    cuts = {'native': matrix}
    distillation = {}
    for rank in (2, 3):
        cuts[f'uniform_r{rank}'] = rank_cut(matrix, None, rank)
        cuts[f'empirical_r{rank}'] = rank_cut(matrix, weights, rank)
        cuts[f'mixed_r{rank}'] = rank_cut(matrix, .5 * weights + .5 * np.ones_like(weights) / weights.size, rank)
        for base_name in (f'uniform_r{rank}', f'empirical_r{rank}'):
            name = f'taskfit_{base_name}'
            cuts[name], distillation[name] = task_cut(model, matrix, bias, train_x[:12000], val_x[:4000],
                                                     rank, cuts[base_name], cut_steps)
    scores = {}
    with torch.no_grad():
        reference = model.from_bits(exact_bits(matrix, bias, test_x)).numpy()
    for name, cut in cuts.items():
        scores[name] = evaluate(model, test_x, test_y, test_attrs, cut, bias)
        scores[name]['uniform_table_relative_mse'] = float(np.sum((cut - matrix)**2) / np.sum(matrix**2))
        with torch.no_grad():
            output = model.from_bits(exact_bits(cut, bias, test_x)).numpy()
        for part, mask in (('all', np.ones(len(test_x), dtype=bool)),
                           ('rare_heldout', test_attrs['heldout'])):
            scores[name][part]['fit_output_relative_mse'] = float(np.sum((output[mask][:, FIT_TASKS] -
                reference[mask][:, FIT_TASKS])**2) / np.sum(reference[mask][:, FIT_TASKS]**2))
            scores[name][part]['held_output_relative_mse'] = float(np.sum((output[mask][:, HELD_TASKS] -
                reference[mask][:, HELD_TASKS])**2) / np.sum(reference[mask][:, HELD_TASKS]**2))
    result = {'seed': seed, 'coding': coding, 'tasks': TASKS,
              'cut_fit_tasks': [TASKS[i] for i in FIT_TASKS],
              'cut_heldout_tasks': [TASKS[i] for i in HELD_TASKS],
              'rare_combination': 'rare query and A=0,B=1,C=0 absent from training/validation',
              'train_n': len(train_x), 'test_n': len(test_x),
              'model_selected_step': selected_step, 'exact_forward_gate': gate,
              'distillation': distillation, 'scores': scores,
              'matrices': {name: cut.tolist() for name, cut in cuts.items()}}
    if not analysis_only:
        torch.save(model.state_dict(), str(stem) + '.pt')
    (HERE / f'{stem.name}.json').write_text(json.dumps(result, indent=2) + '\n')
    print('saved', stem, flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train-steps', type=int, default=2500)
    parser.add_argument('--cut-steps', type=int, default=500)
    parser.add_argument('--analysis-only', action='store_true')
    parser.add_argument('--coding', choices=('sparse', 'dense'), default='sparse')
    args = parser.parse_args()
    run(args.seed, args.train_steps, args.cut_steps, args.analysis_only, args.coding)

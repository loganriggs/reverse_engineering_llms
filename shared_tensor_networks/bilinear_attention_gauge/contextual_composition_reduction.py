"""Compare weight-table and sequence-aware cuts with XOR queries held out.

The native model was trained on XOR. This is a *featurizer* holdout: fitting
and rank selection see only primitive and AND queries; XOR is evaluated later.
"""

import json

import numpy as np
import torch

from dual_joint_function_canonical import HERE
from train_contextual_composition_attention import (
    ContextualCompositionAttention, FAMILIES, N_KEYS, examples)


FIT_QUERIES = tuple(FAMILIES['primitive'] + FAMILIES['and'])


def head_table(model):
    """Exact per-key contribution to each head, indexed by query and token."""
    query = np.repeat(np.arange(16), 16)
    key = np.tile(np.arange(16), 16)
    x = np.column_stack((np.repeat(key[:, None], N_KEYS, axis=1),
                         16+query)).astype(np.int64)
    with torch.no_grad():
        h, _ = model.head_outputs(torch.from_numpy(x))
    return h.numpy().reshape(16, 16, model.heads)/N_KEYS


def exact_head(table, x):
    key = x[:, :N_KEYS].numpy()
    query = x[:, -1].numpy()-16
    return table[query[:, None], key].sum(axis=1)


def evaluate(model, table, x, y, query, projections):
    h = torch.from_numpy(exact_head(table, x)).float()
    with torch.no_grad():
        direct, query_z = model.head_outputs(x[:256])
        gate = float((direct-h[:256]).abs().max())
        assert gate < 3e-5, gate
        query_z = model.embed(x[:, -1])
        native = model.read(torch.cat((h, query_z), dim=1)).squeeze(-1)
        outcome = {}
        for name, projection in projections.items():
            cut_h = h @ torch.from_numpy(projection).float()
            edited = model.read(torch.cat((cut_h, query_z), dim=1)).squeeze(-1)
            sub = {}
            for family, ids in FAMILIES.items():
                mask = torch.from_numpy(np.isin(query, ids))
                sub[family] = dict(n=int(mask.sum()),
                    accuracy=float(((edited[mask] > 0)==y[mask].bool()).float().mean()),
                    native_accuracy=float(((native[mask] > 0)==y[mask].bool()).float().mean()),
                    logit_mse=float((edited[mask]-native[mask]).square().mean()),
                    relative_logit_mse=float((edited[mask]-native[mask]).square().sum()/
                                             native[mask].square().sum().clamp_min(1e-12)),
                    head_relative_mse=float((cut_h[mask]-h[mask]).square().sum()/
                                            h[mask].square().sum().clamp_min(1e-12)))
            outcome[name] = sub
    return outcome


def projector(rows, rank):
    _, _, vh = np.linalg.svd(rows, full_matrices=False)
    basis = vh[:rank].T
    return basis@basis.T


def run():
    torch.set_num_threads(1)
    results = []
    for seed in range(5):
        model = ContextualCompositionAttention()
        model.load_state_dict(torch.load(HERE/f'contextual_composition_attention_seed{seed}.pt',
                                         map_location='cpu', weights_only=True))
        model.eval()
        table = head_table(model)
        fit_x, _, fit_query = examples(20000, 20261700+seed)
        fit_mask = np.isin(fit_query, FIT_QUERIES)
        usage = exact_head(table, fit_x[fit_mask])
        weight_rows = table[list(FIT_QUERIES)].reshape(-1, model.heads)
        all_weight_rows = table.reshape(-1, model.heads)
        matrices = {'weight_fit_queries': weight_rows,
                    'sequence_fit_queries': usage,
                    'weight_all_queries_oracle': all_weight_rows}
        projections = {'native': np.eye(model.heads)}
        for rank in (2, 4, 6, 7, 8):
            for name, rows in matrices.items():
                projections[f'{name}_r{rank}'] = projector(rows, rank)
        test_x, test_y, test_query = examples(30000, 20261800+seed)
        frequency = np.arange(1, 17, dtype=float)
        frequency /= frequency.sum()
        shifted_x, shifted_y, shifted_query = examples(30000, 20261900+seed,
                                                        token_prob=frequency)
        test = evaluate(model, table, test_x, test_y, test_query, projections)
        shifted = evaluate(model, table, shifted_x, shifted_y, shifted_query,
                           projections)
        results.append(dict(seed=seed, test=test, shifted=shifted,
                            table=table.tolist(),
                            projections={k:v.tolist() for k,v in projections.items()}))
        print('seed', seed, 'complete', flush=True)
    summary = {}
    for split in ('test', 'shifted'):
        summary[split] = {}
        for name in results[0][split]:
            summary[split][name] = {}
            for family in FAMILIES:
                summary[split][name][family] = {
                    metric: float(np.mean([r[split][name][family][metric]
                                           for r in results]))
                    for metric in ('accuracy', 'native_accuracy',
                                   'relative_logit_mse', 'head_relative_mse')}
    record = dict(description='Five trained, near-perfect query-conditioned bilinear attention toys. Primitive and AND queries only enter weight/sequence cut fitting; XOR queries are held out from featurizer fitting and all selection. All tasks were in native model training. Exact per-key head table is used for evaluation; projector ranks 2/4/6/7/8. Uniform table rows versus empirical sequence head activations; all-query weight rows are an explicitly leaky oracle. Independent uniform and shifted-token tests.',
                  fit_queries=FIT_QUERIES, heldout_family='xor',
                  results=results, summary=summary)
    (HERE/'contextual_composition_reduction.json').write_text(
        json.dumps(record, separators=(',', ':'))+'\n')
    for name in summary['test']:
        print(name, {f:(round(summary['test'][name][f]['accuracy'], 4),
                        round(summary['test'][name][f]['relative_logit_mse'], 4))
                     for f in FAMILIES})
    return record


if __name__ == '__main__':
    run()

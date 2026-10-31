"""Exact full-rank gauge priors and sample-dependent top-k gates on a toy.

All bases are orthogonal and preserve the complete native head response.
Only primitive and AND query data/edits fit the gauges. XOR is a whole
composition-family holdout for gauge fitting and selection.
"""

import json

import numpy as np
import torch

from contextual_composition_reduction import (
    FIT_QUERIES, exact_head, head_table)
from dual_joint_function_canonical import HERE
from flagship_rank96_exact_gauge_priors import varimax
from train_contextual_composition_attention import (
    ContextualCompositionAttention, FAMILIES, examples)


def orthogonal_random(rng, width):
    q, r = np.linalg.qr(rng.normal(size=(width, width)))
    return q*np.sign(np.diag(r))[None, :]


def fit_gauges(table, train_h, seed):
    rows = table[list(FIT_QUERIES)].reshape(-1, table.shape[-1])
    edits = np.stack([table[q, b]-table[q, a]
                      for q in FIT_QUERIES for a in range(16)
                      for b in range(a+1, 16)])
    _, _, v = np.linalg.svd(train_h, full_matrices=False)
    pca = v.T
    _, _, w = np.linalg.svd(rows, full_matrices=False)
    weight_pca = w.T
    rng = np.random.default_rng(20262000+seed)
    gauges = {'physical_heads': np.eye(table.shape[-1]),
              'sequence_pca': pca, 'weight_pca': weight_pca}
    panels = {'weight_varimax': rows, 'sequence_varimax': train_h,
              'edit_varimax': edits}
    for name, panel in panels.items():
        candidates = []
        for start in (np.eye(table.shape[-1]), *[orthogonal_random(rng, table.shape[-1])
                                                 for _ in range(7)]):
            basis = pca@start
            rotate = varimax(panel@basis)
            chosen = basis@rotate
            z = panel@chosen
            objective = float(np.sum(z**4)-np.sum(np.sum(z*z, axis=0)**2)/len(z))
            candidates.append((objective, chosen))
        gauges[name] = max(candidates, key=lambda pair: pair[0])[1]
    for draw in range(8):
        gauges[f'random_{draw}'] = orthogonal_random(rng, table.shape[-1])
    for basis in gauges.values():
        assert np.max(np.abs(basis@basis.T-np.eye(len(basis))))<1e-10
    return gauges


def evaluate(model, table, x, y, query, gauges):
    h = torch.from_numpy(exact_head(table, x)).float()
    q = model.embed(x[:, -1])
    with torch.no_grad():
        reference = model.read(torch.cat((h, q), dim=1)).squeeze(-1)
        rows = {}
        for name, basis in gauges.items():
            rotation = torch.from_numpy(basis).float()
            z = h@rotation
            for keep in (2, 4, 6):
                selected = z.abs().topk(keep, dim=1).indices
                mask = torch.zeros_like(z).scatter_(1, selected, 1.)
                pruned = (z*mask)@rotation.T
                logits = model.read(torch.cat((pruned, q), dim=1)).squeeze(-1)
                family_rows = {}
                for family, ids in FAMILIES.items():
                    m = torch.from_numpy(np.isin(query, ids))
                    family_rows[family] = dict(
                        n=int(m.sum()),
                        accuracy=float(((logits[m]>0)==y[m].bool()).float().mean()),
                        native_accuracy=float(((reference[m]>0)==y[m].bool()).float().mean()),
                        relative_logit_mse=float((logits[m]-reference[m]).square().sum()/
                                                 reference[m].square().sum().clamp_min(1e-12)),
                        head_relative_mse=float((pruned[m]-h[m]).square().sum()/
                                                h[m].square().sum().clamp_min(1e-12)))
                rows[f'{name}_top{keep}'] = family_rows
    return rows


def run():
    torch.set_num_threads(1)
    results = []
    for seed in range(5):
        model = ContextualCompositionAttention()
        model.load_state_dict(torch.load(HERE/f'contextual_composition_attention_seed{seed}.pt',
                                         map_location='cpu', weights_only=True))
        model.eval()
        table = head_table(model).astype(np.float64)
        train_x, _, train_q = examples(12000, 20262100+seed)
        fit_h = exact_head(table, train_x[np.isin(train_q, FIT_QUERIES)])
        gauges = fit_gauges(table, fit_h, seed)
        test_x, test_y, test_q = examples(20000, 20262200+seed)
        probability = np.arange(1, 17, dtype=float)
        probability /= probability.sum()
        shift_x, shift_y, shift_q = examples(20000, 20262300+seed,
                                             token_prob=probability)
        results.append(dict(seed=seed,
            test=evaluate(model, table, test_x, test_y, test_q, gauges),
            shifted=evaluate(model, table, shift_x, shift_y, shift_q, gauges),
            gauges={name:basis.tolist() for name,basis in gauges.items()}))
        print('seed', seed, 'complete', flush=True)
    summary = {}
    for split in ('test', 'shifted'):
        summary[split] = {}
        for name in results[0][split]:
            summary[split][name] = {family: {
                metric: float(np.mean([r[split][name][family][metric]
                                       for r in results]))
                for metric in ('accuracy','native_accuracy','relative_logit_mse',
                               'head_relative_mse')}
                for family in FAMILIES}
    record = dict(description='Five trained query-conditioned bilinear attention models. Primitive and AND query rows/sequence states/token edits only fit exact orthogonal head gauges. XOR queries are an entirely withheld composition family for featurizer fitting. Per-example top-|coordinate| gating retains 2/4/6 of eight axes; complete gauges preserve native head output exactly. Compare physical heads, weight/sequence PCA, weight/sequence/edit varimax, and eight random rotations. Evaluate fresh uniform and shifted-token sequences.',
                  fit_queries=FIT_QUERIES, heldout_family='xor',
                  results=results, summary=summary)
    (HERE/'contextual_composition_gauge.json').write_text(
        json.dumps(record, separators=(',', ':'))+'\n')
    for name in summary['test']:
        if name.endswith('top4'):
            print(name, {f: round(summary['test'][name][f]['accuracy'], 4)
                         for f in FAMILIES})
    return record


if __name__ == '__main__':
    run()

"""Exact, data-free observable tensor for a causal first-layer bilinear head.

Run on CPU with: python exact_attention_observable.py
The finite-vocabulary example checks function recovery and three distinct
identifiability claims; it does not assert that the learned heads are sparse.
"""

import argparse
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def observable(embedding, qk1, qk2, ov):
    """T[a,b,o] for all query/key token IDs, summed over physical heads.

    qk1/qk2: [heads, d, d]; ov: [heads, d, output].
    Any model-specific attention-score factors belong in qk1/qk2.
    """
    first = np.einsum('ad,hde,be->hab', embedding, qk1, embedding)
    second = np.einsum('ad,hde,be->hab', embedding, qk2, embedding)
    values = np.einsum('bd,hdo->hbo', embedding, ov)
    return np.einsum('hab,hab,hbo->abo', first, second, values)


def direct_output(ids, embedding, qk1, qk2, ov):
    x = embedding[ids]
    y = np.zeros((len(ids), ov.shape[-1]), dtype=np.float64)
    for i in range(len(ids)):
        for j in range(i + 1):
            for h in range(len(qk1)):
                y[i] += ((x[i] @ qk1[h] @ x[j])
                         * (x[i] @ qk2[h] @ x[j])
                         * (x[j] @ ov[h]))
    return y


def table_output(ids, tensor):
    return np.stack([tensor[ids[i], ids[:i + 1]].sum(axis=0)
                     for i in range(len(ids))])


def recover_from_short_sequences(vocab, width, oracle):
    """Recover the whole T using only outputs of lengths one and two.

    This identifies the external *attention contribution* if it can be
    observed separately; it does not identify the internal head tensors.
    """
    diagonal = np.stack([oracle([a])[0] for a in range(vocab)])
    recovered = np.empty((vocab, vocab, width), dtype=np.float64)
    for a in range(vocab):
        for b in range(vocab):
            recovered[a, b] = oracle([b, a])[1] - diagonal[a]
    return recovered


def weighted_rank_cut(matrix, query_probabilities, rank):
    """Best rank-r matrix under a query-row-weighted squared-error law."""
    weight = np.sqrt(query_probabilities)
    left, values, right = np.linalg.svd(weight[:, None] * matrix,
                                       full_matrices=False)
    return ((left[:, :rank] * values[:rank]) @ right[:rank]) / weight[:, None]


def relative_error(matrix, cut, probabilities):
    numerator = np.sum(probabilities[:, None] * (matrix - cut) ** 2)
    denominator = np.sum(probabilities[:, None] * matrix ** 2)
    return float(numerator / denominator)


def run(seed=0):
    rng = np.random.default_rng(seed)
    vocab, d, nheads, output = 31, 4, 2, 3
    embedding = rng.normal(size=(vocab, d)) / np.sqrt(d)
    qk1 = rng.normal(size=(nheads, d, d)) / np.sqrt(d)
    qk2 = rng.normal(size=(nheads, d, d)) / np.sqrt(d)
    ov = rng.normal(size=(nheads, d, output)) / np.sqrt(d)
    tensor = observable(embedding, qk1, qk2, ov)
    oracle = lambda ids: direct_output(ids, embedding, qk1, qk2, ov)
    recovered = recover_from_short_sequences(vocab, output, oracle)
    recovery_error = float(np.max(np.abs(tensor - recovered)))
    assert recovery_error < 1e-12

    sequence_error = 0.0
    for length in (1, 2, 3, 5, 8):
        for _ in range(50):
            ids = rng.integers(vocab, size=length)
            sequence_error = max(sequence_error, float(np.max(np.abs(
                oracle(ids) - table_output(ids, tensor)))))
    assert sequence_error < 1e-12

    matrix = tensor.reshape(vocab, vocab * output)
    left, singular, right = np.linalg.svd(matrix, full_matrices=False)
    rank = int(np.linalg.matrix_rank(matrix, tol=1e-10))
    # Every query enters exactly twice. All heads therefore share the same
    # degree-two symmetric query feature space, dimension d(d+1)/2.
    query_feature_bound = d * (d + 1) // 2
    assert rank <= query_feature_bound
    rank_reconstruction = (left[:, :rank] * singular[:rank]) @ right[:rank]
    rank_error = float(np.max(np.abs(matrix - rank_reconstruction)))
    assert rank_error < 1e-12
    if rank > 1:
        lower = (left[:, :rank - 1] * singular[:rank - 1]) @ right[:rank - 1]
        lower_frobenius = float(np.linalg.norm(matrix - lower))
        assert np.isclose(lower_frobenius, singular[rank - 1], rtol=1e-10)
    else:
        lower_frobenius = float(np.linalg.norm(matrix))

    uniform = np.full(vocab, 1 / vocab)
    common = np.r_[np.full(5, .95 / 5), np.full(vocab - 5, .05 / (vocab - 5))]
    rare = np.r_[np.zeros(5), np.full(vocab - 5, 1 / (vocab - 5))]
    mixed = (uniform + common) / 2
    laws = dict(uniform=uniform, common=common, mixed=mixed, rare=rare)
    cuts = {name: weighted_rank_cut(matrix, law, 3)
            for name, law in laws.items() if name != 'rare'}
    rank_three_error = {name: {law_name: relative_error(matrix, cut, law)
                               for law_name, law in laws.items()}
                        for name, cut in cuts.items()}
    assert rank_three_error['uniform']['uniform'] <= min(
        rank_three_error['common']['uniform'],
        rank_three_error['mixed']['uniform']) + 1e-12
    assert rank_three_error['common']['common'] <= min(
        rank_three_error['uniform']['common'],
        rank_three_error['mixed']['common']) + 1e-12

    # A branch gauge and an extra physical head both preserve the *entire*
    # tensor. Neither can be distinguished by external token sequences.
    scaled_first = qk1.copy(); scaled_second = qk2.copy()
    scaled_first[0] *= 2.7; scaled_second[0] /= 2.7
    branch_gauge_error = float(np.max(np.abs(
        tensor - observable(embedding, scaled_first, scaled_second, ov))))
    split_first = np.concatenate((qk1, qk1[:1]))
    split_second = np.concatenate((qk2, qk2[:1]))
    split_ov = np.concatenate((ov[:1] * .35, ov[1:], ov[:1] * .65))
    split_error = float(np.max(np.abs(
        tensor - observable(embedding, split_first, split_second, split_ov))))
    assert branch_gauge_error < 1e-12 and split_error < 1e-12

    # If a data law omits a query/key cell, that cell can change arbitrarily
    # without affecting its observed pair contributions. The weights select
    # one completion; data on this support alone cannot select it.
    changed = tensor.copy()
    changed[7, 8] += rng.normal(size=output)
    support = [(a, b) for a in range(vocab) for b in range(vocab)
               if (a, b) != (7, 8)]
    seen_error = max(float(np.max(np.abs(tensor[a, b] - changed[a, b])))
                     for a, b in support)
    withheld_error = float(np.linalg.norm(tensor[7, 8] - changed[7, 8]))
    assert seen_error == 0 and withheld_error > 1e-4

    result = dict(seed=seed, vocab=vocab, embedding_width=d,
                  physical_heads=nheads, output_width=output,
                  query_vs_key_output_matrix_rank=rank,
                  symmetric_query_feature_rank_bound=query_feature_bound,
                  singular_values=singular.tolist(),
                  max_short_sequence_recovery_error=recovery_error,
                  max_direct_vs_table_sequence_error=sequence_error,
                  max_rank_reconstruction_error=rank_error,
                  rank_minus_one_frobenius_error=lower_frobenius,
                  rank_three_query_weighted_relative_errors=rank_three_error,
                  max_branch_gauge_tensor_error=branch_gauge_error,
                  max_two_vs_three_head_tensor_error=split_error,
                  observed_support_cell_error=seen_error,
                  withheld_cell_error=withheld_error)
    path = HERE / 'full_width_results' / f'exact_attention_observable_s{seed}.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    run(parser.parse_args().seed)

"""Exact shared-dictionary alternatives for the dense-coded trained toy.

The row-sparse and edit-sparse factorizations implement the same rank-four
query/output-by-key function. The known code is used only for an oracle
counterexample and post-fit scoring, not for either learned gauge.
"""

import json

import numpy as np

from dual_joint_function_canonical import HERE
from train_dense_four_count_weak import SIGNS
from analyze_dense_weak_rule_graph import MEASURES, RAW, graph_scores


def factor_from_rows(matrix, proposed_dictionary):
    """Unit-norm nonorthogonal atoms and exact coefficients within rowspace."""
    _, _, vh = np.linalg.svd(matrix, full_matrices=False)
    space = vh[:4]
    dictionary = (proposed_dictionary @ space.T) @ space
    dictionary /= np.linalg.norm(dictionary, axis=1, keepdims=True)
    if np.linalg.matrix_rank(dictionary, tol=1e-9) != 4:
        raise ValueError('Dictionary has fewer than four independent atoms')
    left = matrix @ np.linalg.pinv(dictionary)
    assert np.max(np.abs(left @ dictionary - matrix)) < 1e-8
    return left, dictionary


def row_sparsity(left, threshold=.1):
    magnitude = np.abs(left)
    relative = magnitude / np.maximum(magnitude.max(axis=1, keepdims=True), 1e-30)
    return dict(active_edges=int((relative >= threshold).sum()),
                active_per_row=relative.__ge__(threshold).sum(axis=1).tolist(),
                normalized_l1=float(np.sum(magnitude / np.maximum(
                    np.linalg.norm(left, axis=1, keepdims=True), 1e-30))))


def bit_edit_purity(dictionary):
    """Post-fit audit: fraction of each atom's edit energy in its top bit."""
    by_bit = []
    for bit in range(4):
        pair = np.array([(t, t ^ (1 << bit)) for t in range(16)
                         if SIGNS[t,bit] == -1])
        changes = dictionary[:, pair[:,1]]-dictionary[:,pair[:,0]]
        by_bit.append(np.mean(changes**2,axis=1))
    energy = np.stack(by_bit, axis=1)
    purity = energy.max(axis=1)/np.maximum(energy.sum(axis=1),1e-30)
    return dict(mean=float(purity.mean()),by_atom=purity.tolist(),
                edit_energy_by_bit=energy.tolist())


def run():
    cuts=np.load(HERE/'dense_weak_rule_reduction_cuts.npz')
    gauge_factors=np.load(HERE/'dense_weak_rule_gauge_factors.npz')
    rows=[]
    for seed in range(10):
        for measure in MEASURES:
            matrix=cuts[f's{seed}_c0.1_r4_{measure}']
            # The q=0 output rows are observable from the function, and are
            # the natural dictionary for a prior on few output-group edges.
            # No raw-bit code enters this construction.
            candidates={
                'output_row_dictionary':factor_from_rows(matrix,matrix[:4]),
                # Oracle: shows the other exact decomposition permitted by
                # the same function and parameter budget. Never fitted.
                'oracle_raw_bit_dictionary':factor_from_rows(matrix,RAW),
                'svd':(gauge_factors[f's{seed}_{measure}_svd_left'],
                       gauge_factors[f's{seed}_{measure}_svd_dictionary']),
                'all_pair_edit_l1':(
                    gauge_factors[f's{seed}_{measure}_all_pair_edits_left'],
                    gauge_factors[f's{seed}_{measure}_all_pair_edits_dictionary']),
                'same_raw_D_bit_edit_l1':(
                    gauge_factors[f's{seed}_{measure}_same_raw_D_bit_edits_left'],
                    gauge_factors[f's{seed}_{measure}_same_raw_D_bit_edits_dictionary'])}
            for name,(left,dictionary) in candidates.items():
                # Scale all bases to unit atom norm. This removes the simple
                # coefficient/dictionary scale escape from sparsity scores.
                lengths=np.linalg.norm(dictionary,axis=1)
                left=left*lengths[None,:]
                dictionary=dictionary/lengths[:,None]
                assert np.max(np.abs(left@dictionary-matrix))<1e-8
                scores=graph_scores(left,dictionary)
                gram=dictionary@dictionary.T
                rows.append(dict(seed=seed,measure=measure,basis=name,
                    max_abs_reconstruction_error=float(np.max(np.abs(
                        left@dictionary-matrix))),
                    row_sparsity=row_sparsity(left),
                    dictionary_condition_number=float(np.linalg.cond(dictionary)),
                    max_pair_atom_cosine=float(np.max(np.abs(
                        gram-np.eye(4)))),
                    edit_purity=bit_edit_purity(dictionary),
                    raw_alignment=scores['mean_raw_atom_alignment'],
                    graph_cell_accuracy_at_0_1=scores['graph']['0.1']['cell_accuracy'],
                    graph_exact_row_fraction_at_0_1=scores['graph']['0.1']['exact_row_fraction']))
    names=sorted(set(x['basis'] for x in rows))
    summary={name:{
        'active_edges_mean':float(np.mean([x['row_sparsity']['active_edges']
                                         for x in rows if x['basis']==name])),
        'normalized_row_l1_mean':float(np.mean([
            x['row_sparsity']['normalized_l1'] for x in rows if x['basis']==name])),
        'edit_purity_mean':float(np.mean([x['edit_purity']['mean']
                                        for x in rows if x['basis']==name])),
        'dictionary_condition_number_mean':float(np.mean([
            x['dictionary_condition_number'] for x in rows if x['basis']==name])),
        'max_pair_atom_cosine_mean':float(np.mean([
            x['max_pair_atom_cosine'] for x in rows if x['basis']==name])),
        'raw_alignment_mean':float(np.mean([x['raw_alignment']
                                         for x in rows if x['basis']==name])),
        'graph_accuracy_mean':float(np.mean([x['graph_cell_accuracy_at_0_1']
                                          for x in rows if x['basis']==name]))}
        for name in names}
    output=dict(description='Exact rank-four shared-dictionary ambiguity. Output-row dictionary constructed from trained function without raw code; raw-bit oracle uses known code only for counterexample and scoring. Edit sparse dictionaries come from prior answer-free gauge fitting.',
                rows=rows,summary=summary)
    (HERE/'dense_weak_group_identifiability.json').write_text(
        json.dumps(output,indent=2)+'\n')
    for name in names:
        print(name,summary[name])


if __name__=='__main__':run()

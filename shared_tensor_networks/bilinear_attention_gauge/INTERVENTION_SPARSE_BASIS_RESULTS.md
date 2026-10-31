# Does intervention-aware compression retain semantic components?

The [intervention-aware key-space experiment](INTERVENTION_DICTIONARY_RESULTS.md) showed that the two-direction ordinary-function space fails QK-branch edits, while four or five directions cover more counterfactuals. That raises a second question: after enlarging the key space, can a **label-free sparsity prior** still produce human-legible components?

For each of ten trained two-head toys, take the right singular key subspace fitted to (a) ordinary head tables only at rank two, (b) ordinary plus both single-branch edits at rank four, or (c) all four ordinary/single/double-edit conditions at rank four or five. Hold the subspace fixed and search orthogonal rotations that minimize the L1 norm of its \(r\times6\) key dictionary. Sixteen deterministic/random starts, pairwise Givens-angle sweeps, and 256 trial angles per pair are used; the best local solution is kept. The rotation changes **neither** the represented key subspace nor any reconstruction error. Task labels and bit-flip responses are used only after optimization.

There are two distinct ways to judge legibility. **Axis cosine** matches individual dictionary rows to the signed A/B token contrasts. **Effect purity** computes each component's response energy under A, B, and filler token flips in the original reduced function, takes the largest fraction, then averages across components weighted by their total bit-flip energy. A high purity does not necessarily mean one component represents a whole bit; two token-specific components can split that bit. The table reports means over ten seeds; random controls average 128 orthogonal rotations per seed.

| Fit and rank | A/B contrast coverage by whole subspace | SVD axis cosine | Sparse axis cosine | SVD effect purity | Random effect purity | Sparse effect purity |
|---|---:|---:|---:|---:|---:|---:|
| Ordinary, rank 2 | .976 | .894 | **.974** | .845 | .820 | **.997** |
| Ordinary + two single edits, rank 4 | .986 | .785 | .719 | .772 | .811 | **.981** |
| All four conditions, rank 4 | .988 | .707 | .721 | .744 | .795 | **.978** |
| All four conditions, rank 5 | .991 | .707 | .718 | .746 | .727 | **.985** |

At rank two, sparse rotation recovers approximately one A direction and one B direction. At rank four, the task contrasts remain in the **subspace** (.986–.988 coverage), and the sparse rotation yields highly selective component effects (.978–.981 purity), but a **single-axis** contrast match falls to about .72. In the rank-four fit to ordinary plus two single edits, all ten seeds have two A-dominant and two B-dominant components above 1% of total bit-flip energy. This is a change in representation granularity: sparsity tends to split a signed bit contrast into token-specific pieces while leaving the task information available in their span. The average number of components with >90% single-bit effect and >1% of total effect is 3.7 under sparse rotation, versus 1.3 under SVD.

The effect-purity gain over both SVD and random rotations is real for this toy, but it is a **post-fit evaluation** based on known bit-flip interventions. It does not mean the sparse objective discovered the A/B grouping without supervision, nor that the physical model computes separate token-specific nodes. A next featurizer should represent *groups of sparse axes* and test whether an unsupervised grouping rule can recover the A/B variables and predict held-out intervention combinations. The causal-fidelity limits from the previous experiment remain: rotating a fixed subspace cannot repair its failure on a held-out double edit.

Run `dual_intervention_sparse_basis.py` after the saved toy checkpoints. `dual_intervention_sparse_basis.json` includes per-seed dictionaries, selected starts, alignment, all component effect energies, random controls, and subspace coverage.

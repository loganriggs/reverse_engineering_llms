## H815 — Explicit shared/private atoms work algebraically, but do not yet improve causal fidelity

Implement eight explicit dictionary atoms: two shared directions C and three private directions per group. Both group memberships include the same two shared atom indices. Fit C as the leading two PCA directions of uniformly weighted, unit-normalized training responses, then fit two rank3 groups to residuals orthogonal to C. Gate by projected energy, selecting one group during causal evaluation. Thus five coordinates are active with eight stored atoms.

This is a staged, constrained shared/private hypothesis, not joint optimization of arbitrary overlapping memberships or learned L0 group/coordinate/membership penalties. Task labels do not determine group fitting or selection. Shared PCA directions are fixed before private fitting. Each private block is orthogonal to C and internally orthonormal; the two private blocks may overlap geometrically. The existing internal lexical split, two tasks, and held-out causal replay remain fixed.

### Union-decoder correctness

Build B=[C,P0,P1] with binary group membership rows selecting C+P0 and C+P1. Encode one global coordinate vector a=B^T x and decode B(a*q), where q marks the union of active groups. Test neither, first, second, and both active. For both groups the expected reconstruction is CC^T x+P0P0^T x+P1P1^T x: the shared contribution appears once. Maximum absolute disagreement is 5.56e-17. Dictionary columns have unit norm and each active five-column block is orthonormal to numerical precision.

When both groups activate, this union decoder is not generally an orthogonal projection onto the combined span because private blocks can overlap. That is a property of the stated shared-coordinate decoder, not an assertion that geometric overlap disappears. The both-active condition is algebraically audited here; causal evaluations activate one group only.

### Storage, active dimension, and causal error

Errors are log-probability RMS reconstruction errors relative to the tested MLP full-versus-zero contribution, not total model response or unique causal shares.

| Model | Stored atoms | Active dimensions | Copy error | IOI error |
|---|---:|---:|---:|---:|
| Flat PCA | 5 | 5 | .294 | .310 |
| Flat PCA | 8 | 8 | .232 | .320 |
| Independent two rank4 groups | 8 | 4 | .273 | .330 |
| Independent two rank5 groups | 10 | 5 | .232 | .310 |
| Shared2 + private3 per group | 8 | 5 | .256 | .335 |

Shared/private improves copying over the same-storage rank4 grouped model but activates one extra coordinate. It uses fewer stored atoms than independent rank5 groups, with worse causal fidelity on both tasks. Flat8 has the same storage but more active coordinates and better copying fidelity. Flat5 uses fewer atoms at the same active dimension and gives better IOI fidelity. No overall Pareto improvement is established.

Shared/private input reconstruction errors are .403 for copying and .528 for IOI. Notably, its copying vector reconstruction is better than independent rank5 groups (.421), yet behavioral error is worse (.256 versus .232). Again reconstruction quality alone is insufficient.

Copying response reversal survives: full swap contrasts -.03400/-.08146 become -.02992/-.06921 at the two doses. IOI full contrasts +.00595/+.01966 become +.00300/+.00981. Both are underestimated. Top-1 accuracy remains 95.8%/100% across all models, so it does not expose the fidelity differences.

### Gate stability

Fit each architecture with the same five random initial partitions (8140–8144), at most fifty iterations, selecting by training reconstruction only. Shared/private selects seed8142 with training normalized loss .11683. Its alternative restarts have held-out assignment agreement of 97.2%,56.9%,91.7%,72.2% after a possible global label swap. The comparable independent rank5 restarts agree only 51.4–62.5%, and rank4 52.8–72.2%. Some shared/private solutions are more stable, but one close-loss solution still yields very different assignments. This is not a robust manifold label recovery claim.

Shared/private training group counts are copy54/42 and IOI11/29; held-out evaluation counts are copy44/52 and IOI10/38, including correlated dose and arm variants. Groups remain mixed rather than identifiable task labels. No hierarchy or implication relation is established.

### Verification and next decisive test

1248 model forwards completed. Full-response historical log-probability replay is exact, paired intervention directions match exactly, train/holdout lexical IDs are disjoint, hashes unchanged, all outputs finite, and basis orthogonality error is 3.11e-15. Training losses are checked monotone within 1e-10, all selected fits converge within eleven iterations, and no undersized group is used. Stability analysis uses saved alternative fits with no additional model forwards. Reserved test untouched; natural-document pool 26,112.

Only three held-out items per task and reused panels support these descriptive comparisons. The explicit sharing is imposed by the architecture; a successful union audit does not show that the model naturally organizes computation this way. Nor is this a complete test of the user's general overlapping-group architecture.

Next keep gates fixed and remove the shared versus selected private contribution separately, then replay the exact RMS/MLP response. Report effects by selected group and task, alongside full and zero controls. This tests whether the proposed shared component actually carries useful causal content for both groups. A positive result would establish operational reuse within this fitted representation, still requiring fresh data and stronger geometry before a manifold or semantic hierarchy claim.

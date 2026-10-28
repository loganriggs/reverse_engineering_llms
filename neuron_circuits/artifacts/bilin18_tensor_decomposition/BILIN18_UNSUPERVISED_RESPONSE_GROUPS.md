## H814 — Label-free response groups offer limited gains and unstable assignments

Fit two four-dimensional response subspaces with hard selection by maximum projected energy. Training uses unit-normalized raw L6 response vectors, uniform per-row weight, and no task labels in the objective or gate. Alternate exact group SVD updates and reassignment, for up to fifty iterations, across five fixed random partitions (8140–8144). Select the lowest training reconstruction loss only; no held-out behavior informs selection. Compare with flat rank4/rank8, task-balanced shared8, and task-oracle rank4 bases under the H812 held-out causal replay protocol.

Uniform rows do not balance lexical/task counts: training has 96 copy and 40 IOI response variants. This is explicit, and uniform flat controls use the same weights. Task-oracle bases and the older task-balanced pooled basis are labeled comparators, not part of the unsupervised group fit.

### Causal fidelity

Errors below are output log-probability RMS error relative to the full-versus-zero tested MLP response contribution, pooled over two doses and own/input-swap arms.

| Representation | Stored columns | Active dimensions | Copy error | IOI error |
|---|---:|---:|---:|---:|
| Uniform flat rank4 | 4 | 4 | .316 | .322 |
| Uniform flat rank8 | 8 | 8 | .232 | .320 |
| Task-balanced flat rank8 | 8 | 8 | .223 | .357 |
| Task-oracle, two rank4 | 8 | 4 | .222 | .331 |
| Unsupervised, two rank4 | 8 | 4 | .273 | .330 |

The automatic groups improve copying fidelity over the active-dimension-matched flat4 control, but use twice its storage. At equal total storage, flat8 is better for copying and approximately similar for IOI. Task-oracle groups are better for copying with the same storage and active dimension. No clear overall advantage for learned grouping is established.

Group input reconstruction errors remain .434 for copying and .522 for IOI. The copying sign reversal survives, but the .05/.10 swap contrasts are -.02709/-.06332 versus full -.03400/-.08146. IOI contrasts are +.00294/+.01218 versus full +.00595/+.01966. Accuracy remains 95.8%/100% across controls and is insensitive to these errors.

### What do the groups select?

Selected restart 8144 has training normalized reconstruction loss .12312. Group0/group1 training assignments are 40/56 for copying and 32/8 for IOI. Held-out reconstruction evaluations give 34/62 for copying and 44/4 for IOI; these counts include correlated dose and response-arm variants, not independent examples. The groups mix tasks rather than reproducing the task labels exactly. Mixed membership alone does not establish semantic sharing.

A separate deterministic refit reproduces all five restart losses to 1e-12. Relative to the selected restart, held-out group-assignment agreement after allowing a global binary label swap is only 70.8%,72.2%,52.8%,55.6% for the other four restarts. The closest competing training loss (.12626 versus .12312) has only 52.8% held-out agreement. Thus these gates are sensitive to initialization; nearly similar reconstruction objectives can define different groupings. This is evidence against treating the present group labels as stable manifold identities, not proof that no meaningful groups exist.

Every restart converged within eight iterations and its objective was monotone nonincreasing within 1e-10. No empty/undersized group was used; each SVD group has at least four rows. Orthogonality error is 3.11e-15. These checks validate the alternating-fit implementation but not the scientific interpretation of its clusters.

### Scope and next architecture test

1104 model forwards completed; full-response historical replay is exact, paired perturbation directions exactly matched, train/holdout items disjoint, all tracked hashes unchanged, and outputs finite. Stability analysis used stored response data and no extra model forwards. Three held-out lexical items per task and previously examined panels limit inference. Reserved test untouched; document pool remains 26,112.

This is a union-of-subspaces baseline with one active group. It has not established a good manifold, canonical dictionary, hierarchical implication, or shared computation. The groups have separate coordinates and may overlap geometrically, but no atom is explicitly represented once and reused.

Next directly test a shared-plus-private dictionary: two shared atoms and three private atoms per group (eight stored atoms, five active dimensions), with geometry-based gates. Compare against two independent rank5 groups (ten stored, five active), the current two rank4 groups (eight stored, four active), and flat controls. Explicitly audit that shared atoms are decoded once if both groups activate. Keep the tradeoff in storage versus active dimension visible, fit using training reconstruction only, and evaluate held-out causal fidelity and restart stability. A shared/private parameterization is a hypothesis to test, not a justification for attaching a taxonomy to unstable groups.

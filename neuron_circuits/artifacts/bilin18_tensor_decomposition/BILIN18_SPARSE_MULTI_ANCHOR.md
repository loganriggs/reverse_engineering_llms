## H1007 — Overlapping task groups beat shuffled groups on old tasks, but not scalar sparsity

Previous turn was progress: H1006 partly recovered old causal fidelity with multi-task anchors and froze the geometry. H1007 now tests actual state-dependent sparse allocation, with no new dictionary or gate fitting.16256forwards complete:256natural references,8960whole-state state/arm evaluations,640full-donor references and6400paired candidate edits.

Dictionary: frozen orthonormal multi-anchor128 with36anchor coordinates plus92mixed-data complement coordinates. Memberships: six historical8atom task groups, one1atom noun group, and sixteen8atom background groups covering all128coordinates. Select fourdistinct groups greedily by previously uncovered squared coordinate energy. Decode the Boolean union once. Two column permutations preserve each group size and the entire overlap matrix. Controls include Top32individual coordinates and, for every group variant, an independent scalar top-k control with exactly that state's group-union size. Paired edits subtract separately encoded donor/recipient reconstructions; no donor-aware support optimization or supplied task labels.

Real-group active counts average29.76old/31.78new/31.76natural, with minimum24on old tasks. Shuffled groups average~31–32. Per-state matching matters: equal group count alone is not equal coordinate budget. Task groups and background blocks are fixed proposed memberships, not learned manifolds.

| Encoder | Old whole KL | New whole KL | Natural whole KL | Old paired relative KL | New paired relative KL |
|---|---:|---:|---:|---:|---:|
| Dense multi128 |0.00326|0.02840|0.02886|0.1219|0.3558|
| ScalarTop32 |0.00392|0.03344|0.03190|0.1770|0.4372|
| Four real groups |0.00445|0.03827|0.03616|0.1944|0.4855|
| Scalar matched to real union |0.00395|0.03347|0.03193|0.1796|0.4370|
| Four shuffled groups1 |0.00602|0.03605|0.03599|0.3929|0.4623|
| Four shuffled groups2 |0.00595|0.03775|0.03580|0.3827|0.4431|

Whole-state KL is native-to-reconstruction output divergence. Paired relative KL is divergence from the full-donor-patched output normalized within cohort by its divergence from native. Primary summaries still cover400oldstates/50passing cohorts,40newstates/fivepassing cells and256naturalstates/64documents. All diagnostics saved. Scalar32 is sparse in128available coordinates, not a dense32dimensional global subspace.

Real task memberships substantially outperform shuffled memberships on old causal contrasts (~0.194versus0.383–0.393paired relative KL), supporting nonrandom structure in the proposed groups. But they underperform their exactlymatched scalar control0.180, and lose to scalar on all three whole-state domains. Realgroups do not beat shuffledgroups on the new contrast or natural whole-state KL. This does not support declaring group sparsity a superior featurizer.

The noun singleton gate is never selected. Under a fixed four-group budget, its background8atom superset weakly dominates its uncovered-energy score; the same direction is nevertheless included for100%of eligible new states through other groups. Gate labels therefore cannot be read as latent semantic absence/presence. This is an explicit selection-bias limitation, not evidence that the noun direction is unused. Likewise a group's activation can reflect high-energy nuisance coordinates. These rules test energy routing, not learned causal relevance.

Scalar selection minimizes Euclidean error in this fixed orthonormal basis at a fixed support count. A group advantage would have to appear in downstream fidelity, stability, causal selectivity or interpretability; none should be inferred from membership alone. Current results show a useful dictionary and some nonrandom task grouping, but not validated manifolds or reusable computation nodes.

Next information-gaining step: measure discovery-only downstream utility of candidate anchor-group-plus-scalar-fill supports at exactly32coordinates, comparing each to plain scalar32. Use independent endpoint encodings and native/full-patch references as appropriate. This distinguishes a selection-rule bottleneck from a membership/capacity bottleneck before training a more complex gate. Any per-example best-choice oracle must be labeled as an oracle and cannot count as deployed performance. Fit later gates only on permitted discovery data, never these reused evaluation controls. Keep geometry frozen and preserve whole-task holdouts.

Audits:16256/16256forwards; supports/bases/source hashes unchanged. Independent greedy-selection replay, Boolean-union-once decoding, group-size/overlap-preserving shuffle checks and exact per-state scalar budget matching pass. Scalar captured energy is at least group captured energy in every state. Independent support prediction replay error0;60whole-state and40paired saved-logit samples replay with zero KLerror and reconstruction error1.11e-16. All summaries/macros independently replayed, all state/paired keys distinct. Identity logits3.96e-5, natural prefix capture relative3.78e-6. No fitting or reserved tests; natural-document pool26,112unchanged. H961pending; broad goal active.

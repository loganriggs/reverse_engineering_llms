## H713 — Replicated coactivation does not yet establish a causal group advantage

H712 established recurring token-and-position-conditioned joint activity in all eight frozen early-layer groups. Test whether their coordinate contributions move downstream outputs toward another natural context more effectively than control subspaces. This tests a possible computational role; it does not measure semantic coherence or task correctness.

Use four hash-selected unordered donor/receiver pairs for each of all eight H711groups on the H712panel. Each pair shares the current token and position, has the anchor among its Top32coordinates in both contexts, and uses different documents. All64documents are distinct across pairs. Test both directions, giving64directed swaps and32pair clusters. Exclude position127so every receiver has an observed next token. Selection does not use downstream model effects.

For each group sum the donor-minus-receiver changes in all eight raw dual coordinates, decoded through the normalized dictionary. This is not a Top32masked reconstruction or a learned group gate. Cap the edit at1%,2.5%,or5%of receiver residual norm, also capping at the full natural group difference so no edit is amplified. Compare equal-norm rank8globalPCA, one fixed random subspace, and Euclidean projection onto the same group's span. Edit only block1's final-token residual output, preserving the persistent value bus. The mean actual dose for the5%cap is4.917%; some full natural differences are smaller than the cap.

Positive donor-KL gain means outputs become more like the donor's native next-token distribution. It does not imply improved correctness. Fit a temperature adjustment to each edited distribution using only the receiver's native logits; the donor is excluded from that fit.

| Residual norm cap | Group coordinate donor-KL gain | PCA gain | Random gain | Same-group projection gain | Temperature-only gain for coordinate edit |
|---|---:|---:|---:|---:|---:|
|1.0%|0.000400|-0.000095|-0.000035|0.000009|0.000474|
|2.5%|0.001074|-0.000177|-0.000068|0.000068|0.001235|
|5.0%|0.002429|-0.000136|-0.000056|0.000323|0.002645|

At the5%cap, the coordinate-minus-PCA gain is0.002565, with a32pair-cluster bootstrap95%interval[-0.000684,0.006345]; coordinate-minus-random is0.002485[-0.001381,0.007358]. Both directions stay together in each resampled pair, with10000resamples. The corresponding intervals also include zero at the smaller doses. These exploratory intervals describe this small panel and are not corrected for trying several contrasts/doses. The same-group projection contrast is also uncertain:0.002105[-0.000036,0.004667].

Temperature-only gain slightly exceeds coordinate gain on average at every dose. This does not mean the edit is merely temperature: at the5%cap, mean KL(edited||native)is0.00006087and KL(edited||temperature-fit)is0.00005523, leaving most of the distribution-shape difference. Temperature can reproduce the donorward scalar score while missing other changes. Thus donorward gain by itself is not a specificity test.

Observed receiver next-token CE changes by-0.001647nats on average at the5%cap for coordinate edits (negative means improvement), versus-0.000410forPCAand+0.001388forrandom. These are tiny changes on one observed continuation per endpoint, not evidence of task competence. Mean KL(native||edited)is0.00006090forcoordinate,0.00006301forPCAand0.00002260forrandom. Matching residual edit norm does not match downstream effect magnitude.

Group effects vary. Group200has the largest coordinate donorward gain0.009174, but PCAon its same pairs gives0.009535. Group121, whose earlier single-coordinate result was dominated by a large confidence-changing outlier, gives-0.000092on these new small-dose group edits. Neither observation justifies selecting a winning group or assigning a semantic label.

Conclusion: the frozen neighborhoods have replicated contextual coactivation, but this experiment does not establish superior causal organization or a meaningful manifold. The alternatives' synthetic recovery/correctness results remain useful, yet the current real-activation pilot is not a verified replacement for the BSF. Representation geometry, contextual covariance, and computational reuse remain separate hypotheses.

Next information-gaining comparison: freeze matched control neighborhoods chosen from directions with similar activation frequencies, decoder geometry and anchor overlap, then test whether the actual coactivation grouping predicts selective intervention effects beyond those controls. Compare anchor-only and neighbor-only contributions, rather than increasing dose or labeling whichever group has the highest donor gain. Any surviving interpretation should then face diverse competence-checked tasks and held-out combinations; retain donorKL, continuationCE and task correctness as separate outcomes. This returns to the user's manifold/computation objective without treating featurizer reconstruction as the answer.

Validation:896whole-model prefix evaluations;64nativeendpoints,64zero-dose replays,768nonzero edits. Prefix-state replay relative error1.79e-6; zero-doseKL8.11e-12; norm matching relative error1.90e-7; group QRorthogonality error3.58e-7. Independent CPUfloat64 reconstruction of every group delta agrees to1.39e-5relative and independently recomputed donor gains to2.27e-14absolute. All input/model/source hashes unchanged, all metrics finite, all8groups included. Original reserved test remains untouched. No new unique research documents; pool remains26112. Broad goal active.

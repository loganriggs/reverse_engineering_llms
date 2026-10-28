## H702 — Staged data-only sparse rotation recovers the synthetic groups

Instead of jointly learning dictionary, groups and gates, first infer sparse directions from activations alone. Fit the8dimensional uncentered PCA span on training inputs, then learn an orthogonal rotation minimizing mean sqrt(coordinate²+1e-6). Orthogonality is enforced by a matrix exponential of a skew-symmetric parameter, so the span reconstruction is preserved while the coordinate basis changes. Three random rotation seeds,3000Adam steps, best checkpoint selected by validation smooth absolute coordinate magnitude. No teacher direction/group/gate labels enter fitting or checkpoint selection.

Infer groups from the training codes: retain examples with at most1%of their energy outside their top3coordinates; count their top3support patterns and take the four most frequent. Group size3and group count4are supplied structural priors. Infer input gates by exact search over16patterns at the original0.01group cost, with1e-5retained as a diagnostic sensitivity setting. Compare the identical group-extraction procedure before learning the rotation.

| Seed | Learned atom cosine | Membership agreement | Eligible training examples | Held group recovery, cost0.01 | Held group recovery, cost1e-5 |
|---|---:|---:|---:|---:|---:|
|61|0.9999975|100.00%|1864|73.93%|98.44%|
|62|0.9999976|100.00%|1864|73.78%|98.97%|
|63|0.9999975|100.00%|1864|73.88%|98.58%|

All three learned rotations recover directions to mean absolute matched cosine above0.999997and exact binary group membership after permutation alignment. Each has1864eligible training examples and four dominant support counts around409–444. Random rotations have only4–20eligible examples, with weak support counts and unreliable groups; one control fills a fourth zero-count pattern deterministically and should not be treated as a supported group. Another random control nearly reconstructs through a constant three-group cover, again showing that reconstruction alone does not establish recovery.

At cost0.01, learned gates recover97.31–97.36%of validation group patterns and73.78–73.93%of held larger combinations; held relative MSE is0.000543–0.000548. At1e-5, held recovery is98.44–98.97%and relative MSE5.50e-7–6.61e-7. The cost sensitivity is consistent with earlier weak-component omission, not a new proof that one penalty is universally correct. These are both reported diagnostic settings; held labels did not choose a checkpoint or cost. The benchmark has been repeatedly explored, so independent teachers/noise conditions are needed for confirmation.

This is the first successful staged label-free recovery in this batch, conditional on strong assumptions: noiseless orthogonal generating directions, known8dimensional span, known4groups of3atoms, sparse singleton/pair training patterns and exact small-group inference. It is not joint neural gate discovery, not a scalable solution to hundreds of groups, and not evidence of superior LLM manifolds. In particular, the orthogonal teacher makes sparse-rotation recovery much easier than a general overlapping nonorthogonal dictionary.

The result changes the next experiment: preserve this as a candidate pipeline and test fresh teachers, observation noise and nonorthogonal directions, with hyperparameters fixed before evaluation. Measure group recovery, reconstruction, support stability and selective learned-component removal separately. Use those results to decide whether this staged approach merits an early-layer activation pilot alongside the existing BSF baseline; retain direct training failures rather than replacing them in the record.

Audits: maximum dictionary orthogonality error9.54e-7; all source hashes unchanged, finite results. Teacher structure is used only for posthoc matching/recovery metrics; group extraction uses training codes only. No original-model forwards/new documents; reserved LLM test split untouched. Broad goal active.

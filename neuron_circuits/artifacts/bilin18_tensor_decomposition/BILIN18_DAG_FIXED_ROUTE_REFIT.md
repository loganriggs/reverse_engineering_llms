## H699 — Equal-output weighting repairs the routed DAG without changing its edges

Warm-start the three H698 sparse routed-readout fits, freezing their learned four-edge route0/1→2→4/5. Keep node gates on and refit continuous functions/readout with fresh Adam for3000steps. Compare the original aggregate normalized squared error with the mean of three per-output squared errors divided by each output's training mean-square target. Within seed, both arms use identical minibatches and initial weights. Select on independent validation under the corresponding objective, including the initial checkpoint; held positive-positive input quadrant is evaluated only before and after selection.

| Seed | Initial held relative MSE | Aggregate-loss refit | Equal-output refit |
|---|---:|---:|---:|
|41|0.155158|0.154835|1.80644e-13|
|42|0.00133434|3.56543e-14|4.04255e-14|
|43|3.01775e-05|1.48727e-14|1.63951e-14|

Seed41's original failure persists under aggregate refitting: independent-output MSE remains3.85. With equal-output weighting it falls to1.54e-14, while both nonlinear outputs also become accurate. All three equal-output fits achieve held relative MSE below1.81e-13. Seeds42/43 are repaired by either loss. The fixed route therefore has sufficient representational capacity; changing topology was unnecessary for this failure.

This matched comparison supports a loss-weighting/optimization explanation. It does not prove a particular local-minimum geometry, a unique internal basis, or that output normalization is universally preferable. The training target has differently scaled outputs2xy,3xy,x; aggregate error can prioritize the larger nonlinear targets. Selection objectives differ between arms, as declared, but every successful final fit is judged by the same held aggregate and per-output metrics. These are warm-start repairs, not a from-scratch reliability estimate.

The nonlinear shared node still affects both nonlinear outputs, while the independent output is untouched by its removal under the imposed routed readout. That selectivity is structurally guaranteed by this readout, so it does not independently prove discovered causal isolation. H698's unrestricted-readout intervention remains the less constrained reuse evidence.

### Where the alternative-featurizer tests now stand

- Shared dictionaries implement union support correctly. Pair coverage and guarded membership search repair conditional synthetic recovery, but joint label-free discovery is not yet demonstrated.
- Gate learning has a large optimization gap. Exact inference, label distillation and expected-cost training expose different tradeoffs between support recovery and reconstruction.
- Hierarchy implementations compose correctly, but ancestor/child mixing can preserve reconstruction while changing component interventions. Residualization selects a convention that works for an orthogonal teacher and changes a nonorthogonal one.
- Computational DAGs can learn sparse shared routes within a supplied candidate graph. Redundant paths and output scaling matter; edge count and accurate prediction alone do not establish the intended internal units.

The highest-priority remaining prerequisite is a joint, label-free shared-dictionary/group/gate experiment from scratch, with oracle-information controls clearly separated and all seeds retained. That closes a more important gap than further tuning this now-repaired DAG. After a bounded attempt, bring practical candidates back to early-layer activations for stability, geometry, intervention specificity and diverse task tests, reporting failures rather than certifying a universally better featurizer. Known synthetic topology, orthogonality or gate labels must not silently carry into claims of LLM discovery.

Audits: held inputs exactly replay H698, initial predictions reproduce previous metrics, all source hashes unchanged, finite results. No original-model forwards/new documents; reserved LLM test split untouched. Broad goal remains active.

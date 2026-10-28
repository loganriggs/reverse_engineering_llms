## H1025 — Destination-query changes dominate the competing-context routing shift

Continuation audit: the preceding repository-review response synthesized existing evidence without a new experiment or authoritative research-state change. Treat it conservatively as no experimental progress. Revalidated H1024 and completed the next available diagnostic here; no process was restarted.

Question: does H1024's failure of a constant head13.0 routing coefficient come from the destination query, source key, or rotary positions? This matters for distinguishing reusable lexical content transformations from context-dependent gates.

Design: first four original discovery country pairs only (47–50), two endpoints and five H1024 templates =40 native prefix sequences. These are opened diagnostic contexts, not fresh validation. Captured both normalized pre-rotary query vectors at the final position, both normalized source key vectors, and the native bfloat16 rotary tables. Recombined baseline/context factors in all eight corners. Shapley allocations average each factor's change across orders, sharing interactions; they are not independent causal fractions or fitted predictor performance. No downstream behavioral interventions or model fitting.

| Context | Mean signed coefficient change from baseline | Query aligned share | Key aligned share | Rotary aligned share |
|---|---:|---:|---:|---:|
| Distance4 |+0.01704|78.7%|~0%|21.3%|
| Distance7 |+0.00861|22.5%|~0%|77.5%|
| Competitor before |+0.10747|96.1%|3.9%|−0.05%|
| Competitor after |+0.10126|80.9%|~0%|19.1%|

Baseline mean coefficient −0.673823; competitor-before −0.566351 and competitor-after −0.572559. Positive change means weaker magnitude of this negative signed coefficient; this attention has no softmax. Aligned share is sum_i(phi_factor_i*delta_i)/sum_i(delta_i^2), across eight endpoint examples per template. The query/key factor includes its native RMS normalization; this experiment does not split pre-normalization direction and norm.

Key near-invariance when extra text follows the source is an expected causal-prefix control, not a discovered semantic invariance. Competitor-before changes the source's prefix and allows a key change, but query change still dominates. Its relative source/destination distance stays one; rotary contribution is tiny, with native quantized tables retained. Absolute score shifts for distance-only templates are much smaller than for competitor templates, so a large percentage there does not imply a large functional effect.

Crucial limit: these templates also change syntax (Unlike / rather than). The results establish query-mediated routing differences for these contexts, not that competing entities specifically cause the change. Exact score-factor recombination is an algebraic diagnosis, not proof that query replacement would preserve other edges or final behavior. Eight endpoints correspond to only four country pairs, with repeated templates; no independent-sample confidence claim.

Audits: native coefficient endpoints replay within2.22e-16absolute under the same float64 dot diagnostic; Shapley allocations sum within2.78e-17. Independent NumPy float64 implementation checks all320 corners within2.37e-8 against native-float32 rotations. Saved captured factors, panel, all corners, scalar scores and allocations. Forty prefixes, zero fits, zero reservedBSFtest examples, zero behavioral interventions. Natural-document pool26112unchanged. H1023 package remains frozen and unmodified.

Research consequence: keep the token-local value map as a reusable operation candidate and put explanatory effort into the destination query/gate, rather than relearning lexical payload directions. However, do not build a supervised competing-country label into a featurizer based on these templates alone. Next use discovery-only matched wording controls (country versus non-country fillers with equal token lengths) to test entity competition against discourse/syntax. Then a finite intervention restricted to the query-controlled source edge can test whether the score diagnosis predicts the isolated cached-path effect. A subsequent distinct-task transfer test is still needed before claiming computation reuse. This bounded diagnostic supports the broader featurizer comparison; it does not establish a manifold or complete the goal.

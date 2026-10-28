## H594 — Retaining the complete discovery span does not rescue orthogonal-projection transfer

Reuse H593's stored context and incoming-response bases exactly, without refitting or selecting from validation outcomes. Project onto all 32 available directions, testing context only, incoming response only, and both, plus full/unchanged references. Total 2,560 interventions. No new random-control runs; H593 supplies those comparisons. Both bases have non-negligible smallest singular values relative to their largest (~0.0162 context, ~0.0180 incoming), so this is not padding with near-zero numerical modes.

| Small-dose cell | Context-only internal response error | Context-only relative margin error | Incoming-only relative margin error | Both-span relative margin error |
|---|---:|---:|---:|---:|
|regular/fronted_preposition|0.0000|0.0002|0.0075|0.0074|
|regular/relative_clause|0.8300|0.7948|0.2122|0.8064|
|irregular/fronted_preposition|0.6020|0.6355|0.1108|0.6254|
|irregular/relative_clause|0.8824|1.1264|0.3155|1.1336|

Full context-span projection nearly reconstructs the discovery context-switch effect, as expected, but remains poor on held-out contexts. Irregular fronted context-only relative margin error is 0.636, and irregular relative-clause error is 1.126. Incoming-response compression remains less damaging in these tests. At half dose the qualitative pattern persists. Because incoming bases were fit on positive small-dose responses, negative and larger-dose responses need not lie exactly in their training span even for discovery nouns.

This rules out the rank1/rank8 cutoff as the sole explanation within this orthogonal-projection family. It does NOT prove the span cannot represent the relevant output response. Orthogonal projection minimizes context-vector reconstruction error; different coefficients inside the same span might produce a much better T(dc,e) or behavioral approximation. Calling this a full-span response oracle would overstate the test.

Next compute a per-example response-optimal least-squares oracle within the fixed context span: for the actual e, solve for coefficients over T(a_j,e) that best reproduce the full T(dc,e), then test those responses in the live model. This deliberately uses the full target response and is a diagnostic bound, not a deployable representation or generalization result. If it succeeds where projection fails, the objective/coordinates are the bottleneck. If substantial local response error remains, broader context coverage is required for internal fidelity in this span; KL or answer-loss equivalence could still be easier. Report coefficient norms and conditioning to detect extrapolation or numerical artifacts.

Audits: stored context/incoming basis tensors exactly match H593; identity <=1.87e-6 relative error; full/unchanged references replay within 1.44e-5; source/basis hashes unchanged and finite metrics. The inherited matched-input-norm audit field is zero because this run contains no new random-control matching, not because an additional control was tested. No new prompts or natural-text documents; FineWeb remains 26,112. Broad goal active and incomplete.

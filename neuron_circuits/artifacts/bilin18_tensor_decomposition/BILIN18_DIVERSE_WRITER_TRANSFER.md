## H581 — Diverse-task transfer favors shared writers with context-dependent downstream routes

H580 made progress by distinguishing internal restoration from downstream effect. This experiment expands beyond copying: 80 lexical pairs across five tasks and two templates, with base, opposite-answer, and answer-preserving nuisance prompts at both endpoints, totaling 320 prompts. No token-validity exclusions. Prompt combinations were composed for this experiment; lexical items overlap previous studies and training novelty is not claimed.

Before inference, lock candidate operations MLP3, MLP6 and Attention5, with neighboring controls MLP2, MLP5, Attention4 and Attention6. Fix sites at final and penultimate tokens across all tasks. These positions are not assumed to have the same syntactic or semantic role. For each recipient, swap the selected writer output from the counterfactual donor, same-answer nuisance donor, or itself. In a separate pass replay only the induced local value change through native routing of L4H0 (MLP2/3) or L7H8 (other operations), with the identity bus fixed. The target head can be only a small route among many.

6,720 paired evaluations give 13,440 intervention-stage forward passes. All items, failed templates and controls are retained. Effects are conditional interventions; whole and isolated paths are not an additive mediation decomposition. Donor values can come from different surface tokens at the fixed site, and nuisance edits vary by task. This is not a matched semantic-feature experiment.

### Baseline competence

Gate: at least four pairs, >=75% both-endpoint candidate correctness and >=50% intended-group full-vocabulary top1, assessed per task/template. No individual failure filtering.

| Task/template | Both-endpoint candidate accuracy | Full-vocabulary top1 | Nuisance top1 | Gate |
|---|---:|---:|---:|---|
|agreement_0|1.000|0.875|0.688|pass|
|agreement_1|1.000|0.438|0.625|fail|
|capital_recall_0|1.000|1.000|1.000|pass|
|capital_recall_1|1.000|0.250|1.000|fail|
|entity_binding_0|1.000|0.938|0.938|pass|
|entity_binding_1|1.000|0.438|0.500|fail|
|sentiment_0|1.000|0.625|0.625|pass|
|sentiment_1|0.000|0.000|0.000|fail|
|sequence_copy_0|1.000|1.000|1.000|pass|
|sequence_copy_1|1.000|1.000|1.000|pass|

All five template-0 tasks pass; only copying passes in template 1. Several failures are continuation-format mismatches: factual recall often prefers “the” before a capital description, binding sometimes prefers “the”, and agreement often prefers punctuation. Sentiment's second template fails even candidate accuracy. These failures limit this next-token test; they do not prove the model lacks the underlying task ability. Their intervention results remain in JSON but do not support competent cross-template transfer claims.

### Candidate-operation effects in competent template 0

Average over all eight pairs, both recipients and both fixed sites. Positive donor-directed margin change means movement toward the opposite answer. Nuisance column reports absolute whole-run margin change, not donor direction. These correlated observations are descriptive, not independent population samples.

| Task | Operation | Whole donor-directed margin | Isolated-value donor-directed margin | Whole nuisance absolute margin |
|---|---|---:|---:|---:|
|agreement|mlp_3|-0.0190|0.0001|0.0700|
|agreement|mlp_6|0.0612|0.0011|0.0896|
|agreement|attn_5|0.4302|0.0016|0.0116|
|capital_recall|mlp_3|0.0318|0.0006|0.0973|
|capital_recall|mlp_6|0.0315|0.0012|0.2515|
|capital_recall|attn_5|0.0206|0.0001|0.3603|
|entity_binding|mlp_3|0.0057|0.0006|0.0977|
|entity_binding|mlp_6|0.0278|-0.0003|0.0915|
|entity_binding|attn_5|0.1129|0.0003|0.0757|
|sentiment|mlp_3|-0.0009|0.0008|0.0236|
|sentiment|mlp_6|0.3474|0.0378|0.0389|
|sentiment|attn_5|0.5690|0.0383|0.0738|
|sequence_copy|mlp_3|0.0047|0.0004|0.0853|
|sequence_copy|mlp_6|0.0332|0.0008|0.0953|
|sequence_copy|attn_5|0.0465|0.0004|0.0593|

Attention5 has larger average donor-directed effects in agreement (0.430) and sentiment (0.569) than in copying (0.047). Yet the isolated L7H8 value path carries only 0.0016 and 0.0383 donor-directed margin respectively. This rejects treating that single copying-derived route as the explanation for the whole operation's cross-task effect at these sites. It is consistent with shared upstream operations whose effects reach output through context-dependent routes. It does not identify a shared feature or rule out other carrier positions.

Neighboring controls matter: Attention4 gives agreement donor movement 0.081, while Attention6 gives 0.017. In sentiment, Attention6 gives 0.193 and MLP5 gives 0.106. In copying, Attention6 (0.074) exceeds selected Attention5 (0.047). Candidate operations are neither uniquely effective nor uniformly ordered. Nuisance effects often exceed counterfactual effects in recall, binding and copying. We therefore do not assign task-semantic labels from these swaps.

Full-output KL and intended-answer-group NLL are saved independently from internal writer/value norms and candidate margins. Group NLL uses the fixed answer set; it is not single-token CE. This batch measures the necessary responses but does not yet fit or validate a response-based canonical representation or whole-task-held-out predictor.

### Refreshed hypothesis set after the basic tests

1. Shared contextual writers support several tasks through different downstream routes.
2. Reader gates select task-relevant directions from broad contextual state.
3. One feature can have several token carriers rather than a unique location.
4. A low-dimensional causal-response representation transfers better than raw-state similarity.
5. Large state variance mainly describes answer-preserving context.
6. Small state directions can dominate a selected readout.
7. Punctuation changes contextual state, not just positional phase.
8. Relative position can still change which contextual directions are read.
9. Some apparent semantic effects are lexical-identity effects.
10. Syntactic roles, rather than token identities, organize transferable carriers.
11. Sentence boundaries alter routing or contextual-state construction.
12. Bilinear MLPs implement context-conditioned local operators rather than fixed feature detectors.
13. Attention and MLP writes cooperate; single-write clamps miss interactions.
14. Residual normalization changes the effective importance of a writer.
15. Persistent identity values and contextual local values support distinct routes.
16. Downstream compensation can hide large upstream state differences.
17. One-token competence gates confuse formatting preferences with underlying knowledge.
18. Apparent low-dimensionality depends strongly on data/task coverage.
19. Off-manifold donor swaps can distort otherwise meaningful representations.
20. Data-equivalent computations admit multiple useful bases, with different equivalence classes under internal, KL and loss criteria.

The leading working explanation is 1–2, with 3, 6, 7 and 12 supported locally by earlier tests. Hypothesis 17 directly explains several observed gate failures. Hypotheses 4 and 20 remain proposals, not established results; 9–11, 13–16 and 19 remain important alternatives or confounds.

### Next discriminating experiments

First repair evaluation coverage: preregister second-template continuations with an unambiguous answer boundary, keep original failed cases, and assess newly composed lexical combinations before interventions. Then freeze candidate operations and compare response signatures across held-out tasks/templates. A response grouping must predict donor-directed changes and KL better than raw-state proximity and neighboring-operation controls. Keep answer loss separate. Test role-aligned versus fixed-position sites to distinguish shared computation from a position artifact. For the strongest transfer, cross writer contribution with downstream reader routing to test hypothesis 1 against a fixed-feature/fixed-route account. Do not call geometric compression an explanation unless these causal predictions transfer.

Audits: self full margin error <=3.82e-6 and isolated <=4.30e-6; finite metrics; source and locked manifest unchanged. An independent 40-case live value/projection-hook audit spanning all task/templates and both target heads/sites agrees with routing reconstruction within 3.79e-5 absolute preprojection error (max relative 2.44e-4). The first attempt stopped at a rotary tensor-slicing error before interventions; corrected axis slicing, unchanged panel/hypotheses, then successful rerun. No new FineWeb documents; tracked total 26,112. Broad goal remains active and incomplete.

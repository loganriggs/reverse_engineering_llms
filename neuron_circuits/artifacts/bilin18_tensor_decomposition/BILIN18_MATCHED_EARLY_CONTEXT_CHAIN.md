## H637 — Cumulative early-MLP context replacement partially closes the actual prompt-response gap

H636 was progress: matched MLP5 context replacements changed responses strongly but left roughly90–101% of the full cross-prompt response gap. H637 tests whether extending the replacement across later early MLPs provides a more explanatory transformation.

Keep the same80 prompts, donor maps and three propagated probes: original, span2, orthogonal2. Cross-template donors preserve exact item identity; within-template donors shift item by four modulo eight, preserving agreement number. Replace only bilinear operating contexts at MLP6–9, or at MLP5–9 together. Normalization, attention, residual paths and final readout remain receiver-native; their tangent inputs change as the intervention propagates. Native operating states are fixed. These are local derivative counterfactuals, not finite patched forward runs or task-accuracy interventions.

The primary outcome is the remaining norm of the donor-versus-receiver full logit-response gap, weighted and centered under receiver-native probabilities. Ratio1 means no improvement; below1 means movement toward the donor response. This is a geometric comparison, not a causal mediation fraction. The donor endpoint includes all its own context-dependent stages and readout, whereas the intervention changes only selected factors.

**Cumulative replacement partially closes the gap.** MLP6–9 context replacement improves213/240 cross-template comparisons; adding MLP5 improves217/240. Mean remaining-gap ratios fall to0.644–0.797 across families for MLP5–9, compared with0.897–1.008 for MLP5 alone. All three probe groups improve on average: original0.664, learned-span probe0.690, orthogonal probe0.854. Improvement is weaker on the orthogonal probe, limiting generality claims.

| Family | Cross: MLP5 only | Cross: MLP6–9 | Cross: MLP5–9 | Within: MLP6–9 | Within: MLP5–9 |
|---|---:|---:|---:|---:|---:|
|copy|0.957|0.805|0.770|0.827|0.805|
|fact|1.008|0.776|0.732|0.976|0.971|
|plural|0.945|0.830|0.737|0.911|0.889|
|agreement|0.907|0.770|0.644|0.778|0.734|
|addition|0.897|0.843|0.797|0.794|0.754|

Within-template controls also improve198/240 comparisons for each cumulative intervention. Agreement and addition show substantial within-template improvement; facts improve little. Therefore donor-context transfer reflects both item and construction variation. A construction label alone is not a sufficient description of the operator. Repeated prompt/probe measurements are not independent trials, and the two donor conditions differ in what they hold fixed.

The intervention remains incomplete: large residual gaps persist in every family. In some cases it worsens the gap. A large mean output-change magnitude does not itself establish explanation; the gap metric is the substantive advance over H636. No claim is made that the task has become more accurate, and competence failures in plurals/addition remain visible.

**Composition matters beyond summing separate changes.** A supplemental audit adds the MLP5-only response change to the MLP6–9-only change, both measured against their native baseline, and compares that additive prediction with the actual combined replacement. This is an interaction of local linear operators, not finite-amplitude neural nonlinearity: replacing an upstream operator changes the tangent entering the replaced downstream operators.

| Family | Additive prediction: remaining cross gap | Actual joint replacement: remaining gap | Interaction norm / original gap norm |
|---|---:|---:|---:|
|copy|0.854|0.770|0.483|
|fact|0.908|0.732|0.581|
|plural|0.829|0.737|0.422|
|agreement|0.689|0.644|0.227|
|addition|0.882|0.797|0.547|

Joint replacement beats the additive prediction on average in every family. Interaction norms are substantial relative to the original gap, but cannot be interpreted as percentages of computation explained: vectors can reinforce or cancel. Additive predictions already improve over the native response on average, so interaction is not required for any improvement at all. This parallels the earlier lesson that accumulation and composition must be separated rather than calling every combined effect irreducible synergy.

The supported working hypothesis is a distributed, context-conditioned sequence of transformations. Operating-context replacement across several MLPs transfers more of the actual response than one MLP alone, including a learned-span and random orthogonal probe. At the same time, incomplete transfer and nonadditive composition argue against reading individual directions or layer effects as independent fixed semantic units. These observations remain local to the sampled model, prompts and injected directions.

Next locate the remaining gap with a controlled boundary at block9. Cross receiver/donor block9 incoming tangents with receiver/donor downstream block10–17 operators and receiver/donor final readouts. Full donor components should reproduce the donor response as a positive control. Mixed combinations can separate information already missing by block9 from later operator and readout differences, with their interactions reported explicitly. Keep the same donor maps and competence audit; use this localization before expanding replacement to more layers indiscriminately.

Audits:no new whole-model forwards;320 cached attention-stage reconstructions;1260 batched JVP calls; native analytic-factor chain reproduces H636 native chain within2.23e-6 relative error and original finite differences within0.001009; all hashes unchanged; finite metrics. Additivity audit uses only existing outputs and is separately hashed. No new natural-text documents; FineWeb remains26,112. Broad goal active and incomplete.

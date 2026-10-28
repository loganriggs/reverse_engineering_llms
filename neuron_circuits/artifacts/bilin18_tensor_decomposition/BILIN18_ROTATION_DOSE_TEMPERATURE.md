## H710 — Confidence control exceeds the strongest scalar donorward gain

Repeat all16directed H708pairs for posthoc-selected directions121/208, using coordinate/PCA/random edits at doses0,0.1,0.25,0.5and1. Save edited logits. For each edit, fit inverse temperature beta in[0.05,20]to the native receiver logits by minimizing KL(Pedited||Ptemperature). The convex one-dimensional optimum is found by matching the edited expectation of the base logits with the temperature distribution's expectation. The donor distribution is not used in this fit. Then measure donorward KL gain for both the actual edit and the fitted temperature control.

Also report entropy and the change in negative log probability of the receiver's observed next token. Positive CE change is worse prediction of that observed token. This is a natural-context continuation metric, not a competence-checked task evaluation.

| Direction | Dose | Mean actual donor KL gain | Mean temperature-control gain | Mean observed next-token CE change |
|---|---:|---:|---:|---:|
|121|0.1|0.003996|0.005702|-0.003920|
|121|0.25|0.017008|0.020723|-0.005271|
|121|0.5|0.058161|0.064341|0.021322|
|121|1|0.102526|0.109870|0.163033|
|208|0.1|-0.004140|-0.003406|-0.000843|
|208|0.25|-0.010126|-0.008519|-0.002056|
|208|0.5|-0.019527|-0.017020|-0.003957|
|208|1|-0.036357|-0.033732|-0.007399|

For the dominant121interview receiver, actual donorward gains across0.1/0.25/0.5/1doses are0.0476/0.1618/0.4833/0.7749; temperature controls gain0.0566/0.1867/0.5391/0.8839. The full edit raises entropy from2.941to4.322nats, with fitted beta0.799. Its top prediction changes from 'What'(native0.494) to 'I'(edited0.271), while 'What'falls to0.106. The observed next token's CE worsens by1.561nats. Smaller doses also do not produce donorward gain beyond the temperature control on this selected case.

The edit is not merely temperature scaling: KL(Pedited||Pbase)is0.994and the best temperature fit leaves0.728. Large token-specific changes remain. What fails is the stronger interpretation that donorward gain demonstrates useful donor-specific content transfer: a donor-independent confidence control achieves greater gain here. The difference between gains is a comparison, not an additive causal decomposition or percent of mechanism explained. KL direction here(edit||base)also differs from H708's receiver KL(base||edit).

Across direction121's eight directed edits, temperature-control mean gain exceeds actual mean gain at every nonzero dose. Average observed next-token CE is slightly improved at0.1/0.25but worsened at0.5/1; the full-dose mean increase is0.163nats. Direction208's small negative donorward gains likewise mostly track a negative temperature-control gain, while its observed next-token CE changes slightly improve. These differing signs reinforce that donorward similarity, predictive loss and useful computation are distinct targets.

Next move from selected scalar effects to training-derived coactivation groups, with token-conditioned controls and held-context repeatability. Test grouped coordinate contributions against rank/norm-matched alternatives and retain dose/confidence controls. Any semantic or discourse interpretation must then be tested on new, diverse, competence-checked prompts rather than inferred from this interview/prose pair. The stable-axis hypothesis is weakened; a multi-direction computational unit remains untested, not vindicated.

Audits:240whole-model prefix evaluations; zero-dose maximum KL2.36e-12; unit-dose donor gains replay H708within2.70e-6; fitted temperature never worse than beta1under its edited-distribution objective. Source hashes unchanged, finite results, test evaluations0, no new documents. Same exploratory selected cases, so no independent confirmation claim. Edited logits retained for further inspection. Broad goal active.

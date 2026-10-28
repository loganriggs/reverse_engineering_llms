## H965 — Separate task directions, with a shared sentiment/negation output direction

Previous goal turn classified as progress: H964 attention localization completed and was independently audited. This turn tests whether the four tasks localized at MLP12 actually share directions. Fit one direction per task using the normalized mean endpoint1-minus-endpoint0 write across four discovery pairs. Freeze before evaluating other cohorts. Project each true donor write difference onto each source-task direction, then intervene with the projection or its complement. This uses the true donor difference to choose the scalar coefficient: it is a teacher-informed span test, not a learned encoder or a standalone prediction.

Discovery direction cosine is0.896 for sentiment versus negation. All other pairwise absolute cosines are below0.026. Cross-task interventions confirm strong sentiment/negation overlap and negligible effects between these and agreement/pronoun directions. Shared localization at layer12 therefore does not imply a universal direction.

KL ratio below means summed KL(full-write donor patch || candidate patch), divided by summed KL(full-write donor patch || native recipient). Zero reproduces the full patch; one is the no-edit baseline. The reference preserves the recipient context and changes its MLP write, avoiding the earlier natural-donor-context confound.

| Target task, lexical candidate cohort | Full-write signed margin effect | Own-direction effect | Own-direction KL ratio | Direct output-label direction KL ratio |
|---|---|---|---|---|
| agreement | 1.338 | 1.311 | 0.248 | 0.470 |
| sentiment | 0.765 | 0.757 | 0.096 | 0.541 |
| negation | 0.903 | 0.874 | 0.016 | 0.458 |
| pronoun_gender | 0.432 | 0.398 | 0.041 | 0.145 |

On lexical candidates, the sentiment direction explains negation with KL ratio0.082 and margin effect0.783 (full0.903). The negation direction explains sentiment with KL ratio0.105 and effect0.730 (full0.765). Thus shared effect goes beyond matching one answer margin: it approximates much of the full-vocabulary distribution change. The privileged direct-label control is the mean unembedding-weight difference between answer groups, normalized before projection; it is not the exact downstream gradient and is not an exhaustive generic-readout control. It is weaker in KL fidelity despite often retaining a large answer-margin effect.

Own-direction removal leaves tiny discovery task effects, but sizeable residual effects can emerge under structural changes. Agreement own-direction effect falls to1.255 versus full1.774 under structural shift; its complement contributes0.545, and own-direction KL ratio0.196. Joint agreement KL ratio0.304. Sentiment transfers better (structural/joint KL0.057/0.104). Negation and pronoun alternate-format native competence fails, so their patch statistics remain diagnostics and are not counted as successful task-generalization evidence.

Crucial confound: sentiment and negation use identical positive/negative output groups, and the current negation panel negates positive adjectives only. This experiment identifies a shared downstream effect direction, not a shared upstream computation, a negation operator, or a general reusable unit. Discovery prototypes use supervised pairing; high fidelity does not demonstrate unsupervised manifold recovery. Four discovery pairs per task and vocabulary overlap restrict generalization claims. Random rank-one and norm-matched paired random interventions are included in the complete JSON.

Audits:2,560 expected and observed whole-model forwards; identity logit max difference3.05e-5; projection orthogonality error1.00e-7 relative; matched-random norm error2.14e-7; finite outputs and unchanged inputs/frozen directions. Independent NumPy direction recovery passed; summary aggregation independently checked. Full-write task effects replay H963 within1.53e-5. No reserved-test or whole-task-holdout access. Natural-document pool remains26,112.

Next: a new fully crossed sentiment-by-negation panel, with positive and negative adjectives each shown both with and without not, and multiple output contexts. Screen native competence before interpreting causal results; keep every failure visible. Frozen H965 directions must be tested without refitting. If edits follow final valence, this is evidence for a shared result representation; if additional context-sensitive components track the negation operation across both polarities, investigate the upstream MLP computation. The H961 frozen context-bridge predictor evaluation remains pending.
